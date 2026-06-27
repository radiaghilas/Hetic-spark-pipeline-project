"""Pipeline data MovieLens : projet jour 4.

Architecture :
    brut (bronze) -> nettoyé (silver, Parquet) -> agrégé (gold, résultats)

Lancement, depuis la racine du projet :
    python starter-code/pipeline.py

L'énoncé complet et la grille : projects/projet-jour-4.md
"""

import sys
import time
import os

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, LongType

from spark_session import get_spark

# Chemins pour MovieLens small
RATINGS_CSV = "data/datasets/ml-latest-small/ratings.csv"
MOVIES_CSV = "data/datasets/ml-latest-small/movies.csv"
SORTIE_SILVER = "data/output/clean"
SORTIE_GOLD = "data/output/analyses"

# Créer le dossier CSV pour l'export
CSV_OUTPUT_DIR = "data/output/analyses/csv"
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)


def ingestion(spark):
    """Étape 1a : lire les données brutes MovieLens.
    
    - ratings.csv : userId, movieId, rating, timestamp
    - movies.csv : movieId, title, genres
    """
    t0 = time.time()
    
    # Schéma explicite pour ratings
    ratings_schema = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("rating", FloatType(), True),
        StructField("timestamp", LongType(), True),
    ])
    
    # Lire ratings
    ratings = spark.read.option("header", "true").schema(ratings_schema).csv(RATINGS_CSV)
    
    # Schéma explicite pour movies
    movies_schema = StructType([
        StructField("movieId", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("genres", StringType(), True),
    ])
    
    # Lire movies
    movies = spark.read.option("header", "true").schema(movies_schema).csv(MOVIES_CSV)
    
    print("\n=== INGESTION ===")
    print(f"Ratings ({RATINGS_CSV}):")
    ratings.printSchema()
    print(f"Nombre de ratings bruts: {ratings.count()}")
    
    print(f"\nMovies ({MOVIES_CSV}):")
    movies.printSchema()
    print(f"Nombre de films: {movies.count()}")
    
    temps = time.time() - t0
    print(f"Temps ingestion: {temps:.2f}s")
    
    return ratings, movies


def nettoyage(ratings, movies, nb_ratings_brut):
    """Étape 1b : nettoyer les données MovieLens (bronze -> silver).
    
    - Valider les ratings (entre 0.5 et 5.0)
    - Valider timestamps
    - Retirer les doublons
    - Retirer les manquants
    """
    t0 = time.time()
    
    print("\n=== NETTOYAGE ===")
    nb_ratings_avant = ratings.count()
    print(f"Ratings avant nettoyage: {nb_ratings_avant}")
    
    # Filtrer les ratings invalides (doivent être entre 0.5 et 5.0)
    ratings_clean = ratings.filter(
        (F.col("rating") >= 0.5) & (F.col("rating") <= 5.0)
    )
    
    # Valider timestamps (doivent être > 0)
    ratings_clean = ratings_clean.filter(F.col("timestamp") > 0)
    
    # Retirer les doublons sur (userId, movieId)
    ratings_clean = ratings_clean.dropDuplicates(["userId", "movieId"])
    
    # Retirer les lignes avec valeurs manquantes
    ratings_clean = ratings_clean.na.drop()
    
    # Nettoyage des movies
    movies_clean = movies.dropDuplicates(["movieId"]).na.drop()
    
    nb_ratings_apres = ratings_clean.count()
    nb_movies = movies_clean.count()
    
    doublons_supprimés = nb_ratings_avant - nb_ratings_apres
    pct_supprimé = (doublons_supprimés / nb_ratings_avant * 100) if nb_ratings_avant > 0 else 0
    
    print(f"Ratings après nettoyage: {nb_ratings_apres}")
    print(f"Doublons/anomalies supprimés: {doublons_supprimés} ({pct_supprimé:.2f}%)")
    print(f"Films uniques: {nb_movies}")
    
    temps = time.time() - t0
    print(f"Temps nettoyage: {temps:.2f}s")
    
    return ratings_clean, movies_clean


def ecrire_silver(ratings_clean, movies_clean):
    """Étape 1c : écrire les couches silver en Parquet."""
    t0 = time.time()
    
    print("\n=== ÉCRITURE SILVER ===")
    
    # Écrire ratings
    ratings_clean.write.mode("overwrite").parquet(f"{SORTIE_SILVER}/ratings")
    print(f"Ratings écrites dans {SORTIE_SILVER}/ratings")
    
    # Écrire movies
    movies_clean.write.mode("overwrite").parquet(f"{SORTIE_SILVER}/movies")
    print(f"Movies écrites dans {SORTIE_SILVER}/movies")
    
    temps = time.time() - t0
    print(f"Temps écriture silver: {temps:.2f}s")


def transformation_et_analyses(spark):
    """Étape 2 : analyses MovieLens (silver -> gold).
    
    Trois analyses :
    1. Agrégation : Films les mieux notés (avec seuil minimum de votes)
    2. Jointure + Broadcast : Statistiques complètes des films
    3. Window Function : Top 5 films par genre (ranking)
    """
    t0 = time.time()
    
    print("\n=== TRANSFORMATION & ANALYSES ===")
    
    # Relire les données silver
    ratings = spark.read.parquet(f"{SORTIE_SILVER}/ratings")
    movies = spark.read.parquet(f"{SORTIE_SILVER}/movies")
    
    # Cache des données réutilisées par plusieurs analyses
    ratings = ratings.cache()
    ratings.count()  # matérialise le cache
    print(f"Cache ratings: {ratings.count()} lignes")
    
    # --- Analyse 1 : Agrégation - Top films par note moyenne ----------
    print("\n--- Analyse 1 : Agrégation (Top films notés) ---")
    t1 = time.time()
    
    min_votes = 20  # Seuil minimum de votes pour éviter les biais
    
    analyse_1 = (ratings
        .groupBy("movieId")
        .agg(
            F.count("rating").alias("nb_votes"),
            F.avg("rating").alias("note_moyenne"),
            F.min("rating").alias("note_min"),
            F.max("rating").alias("note_max"),
            F.stddev("rating").alias("note_stddev")
        )
        .filter(F.col("nb_votes") >= min_votes)
        .join(movies, on="movieId", how="left")
        .select(
            "movieId",
            "title",
            "nb_votes",
            F.round("note_moyenne", 2).alias("note_moyenne"),
            F.round("note_min", 1).alias("note_min"),
            F.round("note_max", 1).alias("note_max"),
            F.round("note_stddev", 2).alias("note_stddev")
        )
        .orderBy(F.desc("note_moyenne"), F.desc("nb_votes"))
    )
    
    print(f"Films avec {min_votes}+ votes trouvés: {analyse_1.count()}")
    analyse_1.show(10)
    temps1 = time.time() - t1
    print(f"Temps analyse 1: {temps1:.2f}s")
    
    # --- Analyse 2 : Jointure avec Broadcast - Stats complètes par film ---
    print("\n--- Analyse 2 : Jointure avec Broadcast (stats films) ---")
    t2 = time.time()
    
    # Broadcast la table des films (petite) dans la jointure
    movies_broadcast = F.broadcast(movies)
    
    analyse_2 = (ratings
        .join(movies_broadcast, on="movieId", how="left")
        .groupBy("movieId", "title", "genres")
        .agg(
            F.count("rating").alias("nb_votes"),
            F.avg("rating").alias("note_moyenne"),
            F.count("userId").alias("nb_utilisateurs_uniques")
        )
        .withColumn("popularite", F.col("nb_votes"))
        .select(
            "movieId",
            "title",
            "genres",
            F.round("note_moyenne", 2).alias("note_moyenne"),
            "nb_votes",
            "nb_utilisateurs_uniques",
            "popularite"
        )
        .orderBy(F.desc("popularite"))
    )
    
    print(f"Statistiques complètes calculées pour {analyse_2.count()} films")
    analyse_2.show(10)
    temps2 = time.time() - t2
    print(f"Temps analyse 2 (avec broadcast): {temps2:.2f}s")
    
    # --- Analyse 3 : Window Function - Top 5 films par genre ---
    print("\n--- Analyse 3 : Window Function (Top 5 par genre) ---")
    t3 = time.time()
    
    # Exploder la colonne genres (un film peut avoir plusieurs genres)
    films_par_genre = (analyse_2
        .withColumn("genre", F.explode(F.split(F.col("genres"), "\\|")))
        .select("movieId", "title", "genre", "note_moyenne", "nb_votes")
    )
    
    # Window function : classement par genre
    fenetre = Window.partitionBy("genre").orderBy(F.desc("note_moyenne"), F.desc("nb_votes"))
    
    analyse_3 = (films_par_genre
        .withColumn("rang", F.row_number().over(fenetre))
        .filter(F.col("rang") <= 5)
        .select(
            "genre",
            "rang",
            "title",
            F.round("note_moyenne", 2).alias("note_moyenne"),
            "nb_votes",
            "movieId"
        )
        .orderBy("genre", "rang")
    )
    
    print(f"Top 5 films par genre calculés")
    analyse_3.show(30)
    temps3 = time.time() - t3
    print(f"Temps analyse 3 (window): {temps3:.2f}s")
    
    temps_total = time.time() - t0
    print(f"\nTemps total analyses: {temps_total:.2f}s")
    
    return {
        "analyse_1_films_mieux_notes": analyse_1,
        "analyse_2_stats_films_complets": analyse_2,
        "analyse_3_top5_par_genre": analyse_3
    }


def ecrire_gold(resultats):
    """Étape 3 : écrire les résultats en Parquet et CSV."""
    t0 = time.time()
    
    print("\n=== ÉCRITURE GOLD ===")
    
    for nom, df in resultats.items():
        # Écrire en Parquet
        chemin_parquet = f"{SORTIE_GOLD}/{nom}"
        df.coalesce(1).write.mode("overwrite").parquet(chemin_parquet)
        print(f"✓ Parquet écrit: {chemin_parquet}")
        
        # Écrire en CSV (pour lecture directe)
        chemin_csv = f"{CSV_OUTPUT_DIR}/{nom}.csv"
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(chemin_csv)
        print(f"✓ CSV écrit: {chemin_csv}")
    
    temps = time.time() - t0
    print(f"Temps écriture gold: {temps:.2f}s")


def main():
    t_start = time.time()
    
    spark = get_spark("Projet Jour 4 - Pipeline MovieLens")
    print("=" * 70)
    print("PIPELINE MOVIELENS - JOUR 4")
    print("=" * 70)
    print("Spark UI disponible sur http://localhost:4040")

    # Étape 1 : ingestion et nettoyage (bronze -> silver)
    ratings_brut, movies_brut = ingestion(spark)
    nb_ratings_brut = ratings_brut.count()
    
    ratings_clean, movies_clean = nettoyage(ratings_brut, movies_brut, nb_ratings_brut)
    ecrire_silver(ratings_clean, movies_clean)

    # Étape 2 : transformation et analyses (silver -> gold)
    resultats = transformation_et_analyses(spark)

    # Étape 3 : finalisation
    ecrire_gold(resultats)

    # Résumé final
    t_total = time.time() - t_start
    print("\n" + "=" * 70)
    print("RÉSUMÉ DU PIPELINE")
    print("=" * 70)
    print(f"Temps total d'exécution: {t_total:.2f}s")
    print(f"Ratings traités: {nb_ratings_brut}")
    print(f"Ratings nettoyés: {ratings_clean.count()}")
    print(f"Films: {movies_clean.count()}")
    print("=" * 70)

    # Garder la session vivante pour explorer la Spark UI
    try:
        input("\nSpark UI sur http://localhost:4040 - Appuyez sur Entrée pour quitter...")
    except EOFError:
        pass
    
    spark.stop()


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        print()
        print("Pipeline incomplet :", e)
        print("Complétez les sections TODO dans starter-code/pipeline.py.")
        sys.exit(1)
