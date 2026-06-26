"""Squelette de pipeline data pour le projet du jour 4.

Complétez les sections marquées TODO avec le jeu de données que vous avez choisi
(taxi NYC multi-mois, DVF immobilier, accidents ONISR, ou MovieLens).

Architecture cible (vue en cours) :
    brut (bronze) -> nettoyé (silver, Parquet) -> agrégé (gold, résultats)

Lancement, depuis la racine du projet :
    python starter-code/pipeline.py

L'énoncé complet et la grille : projects/projet-jour-4.md
"""

import sys
import time
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from spark_session import get_spark

# Chemins. Adaptez-les au jeu de données que vous avez choisi.
# Pour l'option Taxi NYC multi-mois, on lit tous les fichiers mensuels présents.
DATA_BRUT = "data/datasets/yellow_tripdata_2024-*.parquet"
ZONES_CSV = "data/datasets/taxi_zone_lookup.csv"
SORTIE_SILVER = "data/output/clean"
SORTIE_GOLD = "data/output/analyses"

# Créer les dossiers de sortie s'ils n'existent pas
Path("data/output").mkdir(exist_ok=True)


def ingestion(spark):
    """Étape 1a : lire les données brutes.

    TODO :
    - Lire vos données brutes (Parquet : spark.read.parquet ; CSV : spark.read.csv).
    - Pour du CSV, définir un SCHÉMA EXPLICITE (StructType) plutôt que inferSchema :
      plus sûr et plus rapide. Mettre option("sep", ";") pour les CSV français.
    - Inspecter : printSchema(), show(5), count().
    """
    # Lire les 3 mois de données taxi en Parquet (le schéma est embarqué)
    df = spark.read.parquet(DATA_BRUT)

    print("\n=== INGESTION DES DONNÉES BRUTES ===")
    nb_lignes = df.count()
    print(f"Format : Parquet, {nb_lignes:,} lignes")
    print(f"Colonnes : {len(df.columns)}")
    df.printSchema()
    print("\nAperçu (5 premières lignes) :")
    df.show(5)
    return df, nb_lignes


def nettoyage(df, nb_lignes_brut):
    """Étape 1b : typer, dériver des colonnes, nettoyer (bronze -> silver).

    TODO :
    - Créer vos colonnes dérivées avec withColumn (durée, prix au km/m2, heure...).
    - PROTÉGER les divisions : F.when(denominateur > 0, ...).otherwise(None).
    - Filtrer les valeurs aberrantes (montants négatifs, distances/surfaces nulles,
      dates incohérentes). Utiliser & | ~ (pas and/or/not) et parenthéser.
    - Retirer les doublons (dropDuplicates) et gérer les manquants (na.drop/na.fill).
    """
    print("\n=== NETTOYAGE ===")
    
    # 1. Créer des colonnes dérivées
    df = df.withColumn(
        # Durée en minutes
        "duree_min",
        (F.unix_timestamp(F.col("tpep_dropoff_datetime"))
         - F.unix_timestamp(F.col("tpep_pickup_datetime"))) / 60,
    ).withColumn(
        # Prix par km (protéger la division)
        "prix_par_km",
        F.when(F.col("trip_distance") > 0, F.col("fare_amount") / F.col("trip_distance"))
        .otherwise(None),
    ).withColumn(
        # Heure de pickup
        "heure_pickup",
        F.hour(F.col("tpep_pickup_datetime")),
    ).withColumn(
        # Jour de la semaine (1=dimanche, 7=samedi) selon Spark dayofweek
        "jour_semaine",
        F.dayofweek(F.col("tpep_pickup_datetime")),
    ).withColumn(
        # Mois pour traçabilité
        "mois",
        F.month(F.col("tpep_pickup_datetime")),
    )
    
    # 2. Filtrer les valeurs aberrantes
    # Durée entre 0 et 180 minutes (3 heures)
    df = df.filter(
        (F.col("duree_min") > 0) & (F.col("duree_min") <= 180)
    )
    # Distance positive et raisonnable
    df = df.filter(
        (F.col("trip_distance") > 0) & (F.col("trip_distance") <= 100)
    )
    # Montants positifs
    df = df.filter(
        (F.col("fare_amount") > 0) & (F.col("total_amount") > 0)
    )
    # Au moins 1 passager
    df = df.filter(
        F.col("passenger_count") > 0
    )
    
    # 3. Retirer les doublons
    df = df.dropDuplicates()
    
    # 4. Gérer les valeurs manquantes (remplir par 0 pour les colonnes numériques clés)
    df = df.na.fill(
        {"passenger_count": 1, "tip_amount": 0}
    )
    
    nb_lignes_propre = df.count()
    pct_supprime = round(100 * (1 - nb_lignes_propre / nb_lignes_brut), 2)
    
    print(f"Après nettoyage : {nb_lignes_propre:,} lignes")
    print(f"Lignes supprimées : {nb_lignes_brut - nb_lignes_propre:,} ({pct_supprime}%)")
    print("Exemple après nettoyage :")
    df.select("tpep_pickup_datetime", "duree_min", "trip_distance", "fare_amount", 
              "heure_pickup", "jour_semaine").show(5)
    
    return df, nb_lignes_propre


def ecrire_silver(df, spark):
    """Étape 1c : écrire la couche intermédiaire nettoyée en Parquet.

    TODO :
    - Écrire en Parquet (write.mode("overwrite").parquet(SORTIE_SILVER)).
    - Optionnel : partitionBy sur une colonne à FAIBLE cardinalité (mois, département,
      année). Jamais sur une colonne à forte cardinalité (cela crée trop de fichiers).
    """
    print("\n=== ÉCRITURE COUCHE SILVER (Parquet nettoyé) ===")
    # Écrire partitionné par mois (faible cardinalité = 3 mois)
    df.write.mode("overwrite").partitionBy("mois").parquet(SORTIE_SILVER)
    print(f"✓ Couche silver écrite dans {SORTIE_SILVER}")
    print(f"  Partitionnée par mois (3 partitions)")
    
    # Vérifier ce qui a été écrit
    df_verify = spark.read.parquet(SORTIE_SILVER)
    print(f"  Vérification : {df_verify.count()} lignes relues du Parquet")


def transformation_et_analyses(spark):
    """Étape 2 : relire le propre, puis 3 analyses (silver -> gold).

    On relit la couche Parquet nettoyée (pas les données brutes).

    TODO : produire AU MOINS TROIS analyses, dont :
    - une AGRÉGATION (groupBy + agg) ;
    - une JOINTURE (join, idéalement avec F.broadcast sur la petite table) ;
    - une WINDOW FUNCTION (Window.partitionBy(...).orderBy(...), row_number/rank/lag).
    Et au moins UNE OPTIMISATION justifiée : broadcast, cache, ou repartition.
    """
    print("\n=== ANALYSES (silver -> gold) ===")
    
    # Relire la couche Parquet nettoyée
    df = spark.read.parquet(SORTIE_SILVER)
    
    # OPTIMISATION : cache car df sera réutilisé par plusieurs analyses
    print("Cache du DataFrame silver (réutilisé 3 fois)...")
    df = df.cache()
    df.count()  # matérialise le cache
    
    # Charger la table des zones
    zones = spark.read.option("header", "true").option("inferSchema", "true").csv(ZONES_CSV)
    zones = zones.select("LocationID", "Zone", "Borough")
    print(f"Table zones chargée : {zones.count()} zones")
    
    # ========== ANALYSE 1 : AGRÉGATION ==========
    # Revenu total et moyen par zone de pickup
    print("\n--- Analyse 1 : Revenu par zone (AGRÉGATION) ---")
    analyse_1 = (
        df.groupBy("PULocationID")
        .agg(
            F.count("*").alias("nb_courses"),
            F.sum("total_amount").alias("revenu_total"),
            F.avg("total_amount").alias("revenu_moyen"),
            F.avg("tip_amount").alias("pourboire_moyen"),
        )
        .filter(F.col("nb_courses") >= 10)  # au moins 10 courses
        .orderBy(F.desc("revenu_total"))
        .limit(20)
    )
    
    # Joindre avec les noms de zones pour améliorer la lisibilité
    # OPTIMISATION : broadcast la petite table des zones (265 lignes)
    analyse_1 = (
        analyse_1
        .join(
            F.broadcast(zones.select("LocationID", "Zone", "Borough")),
            F.col("PULocationID") == F.col("LocationID"),
            "left",
        )
        .select("PULocationID", "Zone", "Borough", "nb_courses", 
                "revenu_total", "revenu_moyen", "pourboire_moyen")
        .orderBy(F.desc("revenu_total"))
    )
    
    print("✓ Top 20 zones par revenu :")
    analyse_1.show(5)
    
    # ========== ANALYSE 2 : JOINTURE ==========
    # Top 10 trajets (zone de pickup → zone de dropoff) par nombre de courses
    print("\n--- Analyse 2 : Trajets zone-à-zone (JOINTURE + BROADCAST) ---")
    
    # Charger 2 fois la table zones (pour PULocationID et DOLocationID)
    zones_pu = zones.select("LocationID", F.col("Zone").alias("Zone_PU"), 
                            F.col("Borough").alias("Borough_PU"))
    zones_do = zones.select("LocationID", F.col("Zone").alias("Zone_DO"), 
                            F.col("Borough").alias("Borough_DO"))
    
    # Agrégation avant la jointure pour réduire le volume
    trajets = (
        df.groupBy("PULocationID", "DOLocationID")
        .agg(
            F.count("*").alias("nb_courses"),
            F.sum("total_amount").alias("revenu"),
            F.avg("trip_distance").alias("distance_moy"),
        )
        .filter(F.col("nb_courses") >= 10)
    )
    
    # Jointures avec BROADCAST des petites tables
    analyse_2 = (
        trajets
        .join(
            F.broadcast(zones_pu),
            F.col("PULocationID") == F.col("LocationID"),
            "left",
        )
        .drop("LocationID")
        .join(
            F.broadcast(zones_do),
            F.col("DOLocationID") == F.col("LocationID"),
            "left",
        )
        .drop("LocationID")
        .select(
            "PULocationID", "DOLocationID",
            "Zone_PU", "Borough_PU",
            "Zone_DO", "Borough_DO",
            "nb_courses", "revenu", "distance_moy"
        )
        .orderBy(F.desc("nb_courses"))
        .limit(10)
    )
    
    print("✓ Top 10 trajets zone-à-zone par volume :")
    analyse_2.show(5)
    
    # ========== ANALYSE 3 : WINDOW FUNCTION ==========
    # Top 10 heures de la journée par revenu total (classement)
    print("\n--- Analyse 3 : Heures de pickup par revenu (WINDOW FUNCTION) ---")
    
    revenu_par_heure = (
        df.groupBy("heure_pickup", "jour_semaine")
        .agg(
            F.sum("total_amount").alias("revenu_total"),
            F.count("*").alias("nb_courses"),
            F.avg("total_amount").alias("montant_moyen"),
        )
    )
    
    # Fenêtre pour classer par jour de semaine
    window_spec = Window.partitionBy("jour_semaine").orderBy(F.desc("revenu_total"))
    
    analyse_3 = (
        revenu_par_heure
        .withColumn("rang", F.row_number().over(window_spec))
        .filter(F.col("rang") <= 5)  # Top 5 heures par jour de semaine
        .select("jour_semaine", "heure_pickup", "rang", "revenu_total", 
                "nb_courses", "montant_moyen")
        .orderBy("jour_semaine", "rang")
    )
    
    print("✓ Top 5 heures par jour de semaine (classement par Window) :")
    analyse_3.show(10)
    
    # ========== ANALYSE 4 (BONUS) : DÉTECTION D'ANOMALIES ==========
    # Identifier les prix suspects (Z-score > 3 = 3 écarts-types au-delà de la moyenne)
    print("\n--- Analyse 4 (BONUS) : Détection d'anomalies de prix (STATS) ---")
    
    stats_prix = df.select(
        F.avg("total_amount").alias("prix_moyen"),
        F.stddev("total_amount").alias("prix_stddev"),
        F.min("total_amount").alias("prix_min"),
        F.max("total_amount").alias("prix_max"),
        F.percentile_approx("total_amount", 0.99).alias("prix_p99"),
    )
    
    stats_dict = stats_prix.collect()[0].asDict()
    prix_moyen = stats_dict["prix_moyen"]
    prix_stddev = stats_dict["prix_stddev"]
    
    # Calculer le Z-score pour chaque course
    analyse_4 = (
        df.withColumn(
            "z_score",
            (F.col("total_amount") - prix_moyen) / prix_stddev
        )
        .filter(F.col("z_score").cast("int") > 3)  # Anomalies majeures
        .select("tpep_pickup_datetime", "PULocationID", "DOLocationID", 
                "total_amount", "trip_distance", "z_score")
        .orderBy(F.desc("z_score"))
        .limit(20)
    )
    
    print(f"✓ Statistiques de prix :")
    print(f"  Moyenne : ${prix_moyen:.2f}")
    print(f"  Écart-type : ${prix_stddev:.2f}")
    print(f"  Min/Max : ${stats_dict['prix_min']:.2f} / ${stats_dict['prix_max']:.2f}")
    print(f"  P99 : ${stats_dict['prix_p99']:.2f}")
    print(f"\nTop 20 courses avec prix anormal (|Z-score| > 3) :")
    analyse_4.show(5)
    
    # Vérifier que tout est rempli
    if analyse_1 is None or analyse_2 is None or analyse_3 is None or analyse_4 is None:
        raise RuntimeError("L'une des analyses n'a pas pu être créée.")
    
    return {"analyse_1_revenu_zones": analyse_1, 
            "analyse_2_trajets_zones": analyse_2, 
            "analyse_3_heures_semaine": analyse_3,
            "analyse_4_anomalies_prix": analyse_4}


def ecrire_gold(resultats):
    """Étape 3 : écrire les résultats de synthèse.

    TODO :
    - Écrire chaque résultat (Parquet ou CSV). coalesce(1) est acceptable ICI car les
      résultats agrégés sont PETITS. Ne jamais coalesce(1) un gros DataFrame.
    """
    print("\n=== ÉCRITURE COUCHE GOLD (Résultats synthèse) ===")
    
    # Créer les dossiers de sortie
    Path(SORTIE_GOLD).mkdir(exist_ok=True)
    csv_output = Path(SORTIE_GOLD) / "csv"
    csv_output.mkdir(exist_ok=True)
    
    for nom, df in resultats.items():
        chemin_parquet = f"{SORTIE_GOLD}/{nom}"
        chemin_csv = str(csv_output / f"{nom}.csv")
        
        nb_lignes = df.count()
        print(f"✓ Écriture de {nom} ({nb_lignes} lignes)...")
        
        # Écrire en Parquet
        df.coalesce(4).write.mode("overwrite").parquet(chemin_parquet)
        
        # Écrire aussi en CSV (plus lisible)
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(chemin_csv)
        print(f"  ✓ Parquet : {chemin_parquet}")
        print(f"  ✓ CSV : {chemin_csv}")
    
    print("\n✓ Tous les résultats écrits en Parquet et CSV")


def main():
    temps_debut = time.time()
    
    spark = get_spark("Projet Jour 4 - Pipeline Taxi NYC")
    print("\n" + "="*70)
    print("PIPELINE ETL + ANALYSE - TAXI NYC (3 mois)")
    print("="*70)
    print(f"\n✓ Spark UI disponible sur http://localhost:4040")
    print("  (Ouvrez-la pendant un job pour voir le DAG et les stages)\n")

    # Étape 1 : ingestion et nettoyage (bronze -> silver)
    print("\n🔄 ÉTAPE 1 : INGESTION ET NETTOYAGE")
    temps_etape1 = time.time()
    brut, nb_brut = ingestion(spark)
    propre, nb_propre = nettoyage(brut, nb_brut)
    ecrire_silver(propre, spark)
    temps_etape1 = time.time() - temps_etape1

    # Étape 2 : transformation et analyses (silver -> gold)
    print("\n🔄 ÉTAPE 2 : TRANSFORMATIONS ET ANALYSES")
    temps_etape2 = time.time()
    resultats = transformation_et_analyses(spark)
    temps_etape2 = time.time() - temps_etape2

    # Étape 3 : finalisation
    print("\n🔄 ÉTAPE 3 : FINALISATION")
    temps_etape3 = time.time()
    ecrire_gold(resultats)
    temps_etape3 = time.time() - temps_etape3

    temps_total = time.time() - temps_debut
    
    # Résumé final
    print("\n" + "="*70)
    print("✓ PIPELINE TERMINÉ AVEC SUCCÈS !")
    print("="*70)
    print(f"\n📈 Résumé d'exécution :")
    print(f"  Étape 1 (Ingestion + Nettoyage) : {temps_etape1:.2f}s")
    print(f"  Étape 2 (Analyses) : {temps_etape2:.2f}s")
    print(f"  Étape 3 (Export) : {temps_etape3:.2f}s")
    print(f"  ────────────────────────────────")
    print(f"  Total : {temps_total:.2f}s ({temps_total/60:.2f}m)")
    print(f"\n  Données traitées : {nb_brut:,} lignes brutes → {nb_propre:,} lignes propres")
    print(f"  Compression : {100 - (nb_propre/nb_brut)*100:.1f}% de doublons/anomalies")
    
    # Garder la session vivante pour explorer la Spark UI.
    input("\n📊 Spark UI sur http://localhost:4040 - Appuyez sur Entrée pour quitter...")

    spark.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
