# Rapport de projet - Pipeline Spark (Jour 4)

- **Équipe** : Groupe 16
- **Jeu de données** : MovieLens small
- **Date** : 27 juin 2026

- **Archive de livrables** : https://drive.google.com/file/d/1v2qTtXifJHLQG7rMzsvrfH1yU-U6CDz6/view
- **Livrables** : data/output/
- **Capture Spark UI** : Projet Jour 4 - Pipeline MovieLens - Spark Jobs.html

---

## 1. Jeu de données et schéma cible

- Source et volume : fichier CSV MovieLens small, avec 100 836 ratings et 9 742 films.
- Schéma cible retenu :
  - `ratings`: `userId` (int), `movieId` (int), `rating` (float), `timestamp` (long)
  - `movies`: `movieId` (int), `title` (string), `genres` (string)
- Questions métier visées :
  - quels films ont les meilleures notes moyennes avec un seuil de votes suffisant ?
  - quels films sont les plus populaires et les mieux couverts par les utilisateurs ?
  - quels sont les meilleurs films par genre selon un classement window function ?

---

## 2. Pipeline (bronze -> silver -> gold)

```text
brut (bronze)  ->  nettoyé (silver, Parquet)  ->  agrégé (gold)
```

- Nettoyage appliqué : filtres sur `rating` entre 0.5 et 5.0, contrôle du timestamp positif, suppression des doublons sur `(userId, movieId)`, suppression des lignes avec valeurs manquantes, déduplication des films par `movieId`.
- Lignes brutes : 100 836 ratings | après nettoyage : 100 836 | écartées : 0 (0.00 %).
- Partitionnement de la silver : écriture simple en Parquet sous `data/output/clean` ; le choix a été de conserver une couche stable, lisible par Spark, avant les agrégations.

---

## 3. Analyses

### Analyse 1 - agrégation

- Question : trouver les films les mieux notés, en limitant le biais des films très peu notés.
- Code clé :

```python
analyse_1 = (
    ratings.groupBy("movieId")
    .agg(F.count("rating").alias("nb_votes"), F.avg("rating").alias("note_moyenne"))
    .filter(F.col("nb_votes") >= 20)
    .join(movies, on="movieId", how="left")
    .orderBy(F.desc("note_moyenne"), F.desc("nb_votes"))
)
```

- Résultat (extrait) :

```text
movieId | title | nb_votes | note_moyenne
1104    | Streetcar Named Desire, A (1951) | 20 | 4.48
318     | Shawshank Redemption, The (1994) | 317 | 4.43
```

- Lecture métier : les meilleurs films de cette vue sont ceux qui combinent une forte note moyenne et un volume de votes suffisant.

### Analyse 2 - jointure

- Question : compléter les statistiques par film avec les métadonnées de la table `movies` et mesurer la popularité réelle.
- Code clé :

```python
analyse_2 = (
    ratings.join(F.broadcast(movies), on="movieId", how="left")
    .groupBy("movieId", "title", "genres")
    .agg(F.count("rating").alias("nb_votes"), F.avg("rating").alias("note_moyenne"))
)
```

- Résultat (extrait) :

```text
Forrest Gump (1994) | 329 votes | note moyenne 4.16
Shawshank Redemption, The (1994) | 317 votes | note moyenne 4.43
```

- Lecture métier : cette analyse met en évidence les films les plus consultés et les plus appréciés, sans perdre la dimension genre.

### Analyse 3 - window function

- Question : classer les meilleurs films par genre, au sein de chaque catégorie.
- Code clé :

```python
films_par_genre = (
    analyse_2.withColumn("genre", F.explode(F.split(F.col("genres"), "\\|")))
)
fenetre = Window.partitionBy("genre").orderBy(F.desc("note_moyenne"), F.desc("nb_votes"))
analyse_3 = films_par_genre.withColumn("rang", F.row_number().over(fenetre))
```

- Résultat (extrait) :

```text
genre | rang | title | note_moyenne | nb_votes
Action | 1 | The Big Bus (1976) | 5.0 | 1
Drama | 1 | ...
```

- Lecture métier : la vue par genre donne un classement plus fin, mais elle met aussi en lumière des films très peu notés, ce qui limite l’interprétation.

---

## 4. Optimisation

- Optimisation choisie : `broadcast` sur la petite table `movies` dans la jointure avec `ratings`.
- Pourquoi : `movies` contient 9 742 lignes, contre 100 836 lignes dans `ratings`; la rejoindre avec un broadcast évite de redistribuer la petite table sur tous les workers.
- Mesure observée :

```text
join standard : 2.269s
join broadcast : 0.443s
speedup : 5.12x
```

- Ce que ça change : la jointure devient nettement plus rapide, avec moins de mouvement de données et un plan plus favorable pour le moteur Spark.

---

## 5. Lecture de la Spark UI

- Job observé : exécution complète du pipeline, puis les jobs d’agrégation et de window ranking.
- Où se produit le shuffle : dans les stages de `groupBy` et de `window` ; c’est visible dans le DAG et les stages de la Spark UI sur `http://localhost:4040`.
- Nombre de stages et de tasks : la page exportée de la Spark UI permet d’observer les jobs exécutés et leur succession dans le temps ; l’export sauvegardé montre notamment une exécution de l’application avec 87 jobs terminés.
- Capture(s) : voir le fichier Projet Jour 4 - Pipeline MovieLens - Spark Jobs.html pour consulter la Spark UI exportée.
- Commentaire : le pipeline est simple et lisible, mais il reste sensible au shuffle sur les agrégations, ce qui justifie l’optimisation par broadcast.

---

## 6. Exploration au-delà du cours

- Piste choisie : AQE et nombre de partitions sur une agrégation Spark.
- Question : le réglage du nombre de partitions de shuffle change-t-il le temps de calcul d’une simple agrégation sur les ratings ?
- Protocole : même jeu de données, même agrégation `groupBy("movieId")`, seule la valeur de `spark.sql.shuffle.partitions` change ; l’AQE reste activé.
- Mesures :

```text
partitions=2  -> 0.403s
partitions=8  -> 0.226s
partitions=16 -> 0.199s
partitions=32 -> 0.179s
```

- Conclusion : sur ce volume, une valeur plus élevée que la valeur par défaut améliore légèrement le temps de calcul, probablement parce qu’elle évite un trop grand déséquilibre entre les tasks et offre un meilleur parallélisme de shuffle.

---

## 7. Ce qu'on a appris et limites

- Ce qui a marché : l’ingestion avec schéma explicite, la couche silver propre en Parquet, les trois analyses distinctes et la lecture métier associée.
- Ce qui a bloqué : la lecture de la Spark UI est utile, mais une capture n’est pas toujours visible automatiquement dans le dépôt ; il faut la récupérer au moment de l’exécution.
- Ce qu’on ferait avec plus de temps : ajouter une exploration plus poussée sur la répartition des données (skew) ou tester une version avec `spark-submit` pour comparer un run local et un run soumis.
