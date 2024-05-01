Réalisez une analyse de sentiments grâce au Deep Learning
Notre projet est de créer une application de Machine Learning capable de prédire le sentiment des tweets.
La société Air Paradis aimerait pouvoir prédire les "bad buzz".

Pour se faire nous entrainons plusieurs modèles :

Nous commençons par un modèle de régression logistique, nous effectuons plusieurs tests d'hyper paramètres et nous suivons les performances sur MLflow à travers la plateforme Ngrok.
Pour le pré-processing du texte, nous appliquons deux techniques : la lemmatisation et le stemming.

Nous entrainons par la suite deux modèles LSTM, le premier avec Word2Vec et lemmatisation et le deuxième avec GLOVE et stemming.

Nous terminons par un modèle avancé : BERT pré entrainé utilisant la bibliothèque transformers de Hugging Face.

Le modèle BERT est entrainé sur le cloud car il requiert la puissance d'un GPU performant. Nous testons deux configurations : une avec gel des couches pré entrainées et une autre sans gel.
Nous validons l'expérience avec gel car elle est plus cohérente et n'engendre pas d'overfitting.

Le modèle final est entrainé sur 500000 observations et est par la suite déployé sur le cloud avec Azure APP Service.

L'entrainement de BERT n'a pas été suivi avec MLflow car notre choix de départ (utiliser Ngrok) s'est avéré incompatible avec une exécution sur le cloud.

L'API est créée, un fichier index.html est ajouté pour l'interface graphique et les tests unitaires sont exécutés en local.

Le déploiement se fait avec les "actions" de Github, les tests unitaires sont ajoutés au workflow de Github. Le modèle "pytorch_model.bin" est chargé du blob où il est stocké.
Le workflow lance le container, charge le modèle et les dépendances et déploie avec succès notre application "predict-app-api".

Nous utilisons Azure Application Insights pour le suivi et l'amélioration continue de notre modèle.
Nous avons défini une « alert rules » qui déclenche une alerte si dans un intervalle de 60 secondes, 3 mauvaises prédictions sont signalées.
Un mail est alors envoyé.

Outre ce présent repository qui a servi exclusivement à versionner le code de nos modèles, nous avons mis en place un deuxième repository pour les besoins du déploiement et des test unitaires.
Voici le lien vers ce deuxième repository : https://github.com/DhekerKacem/sentiment-analysis
