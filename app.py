from flask import Flask, render_template, request, jsonify
import json

app = Flask(__name__)

# Données du portfolio
PROFIL = {
    'nom': 'Mohamed-El-Moctar Ahmed',
    'titre': 'Data Scientist',
    'email': 'mokhtarkhalil655@gmail.com',
    'linkedin': 'https://www.linkedin.com/in/mohamed-el-moctar-ahmed-009550218/',
    'github': 'https://github.com/mokhtar-khalil',
    'bio': """Data Scientist passionné avec plusieurs expériences en stage dans 
    le machine learning, deep learning, recommendations systèmes et l'analyse de données. Experimenté en Python, SQL et outils de Big Data."""
}

COMPETENCES = {
    'Langages': ['Python', 'R', 'SQL', 'PHP'],
    'Machine Learning & Deep Learning & AI' : ['Scikit-Learn', 'TensorFlow', 'PyTorch', 'XGBoost', 'MlOps', 'Annoy', 'Pickle',
    'OpenAI SDK', 'CrewAI', 'autogen', 'LangGraph', 'LlamaIndex', 'BERT', 'Transfomers'],
    'Data Analysis': ['Pandas', 'NumPy', 'SciPy', 'Statsmodels'],
    'Visualisation': ['Matplotlib', 'Seaborn', 'Plotly', 'Power BI',"Bokeh"],
    'Big Data': ['Apache Spark', 'Hadoop', 'Databricks'],
    'Bases de données': ['PostgreSQL', 'MongoDB', 'MySQL', 'NoSQL'],
    'Web': ['Flask', 'Django', 'FastAPI',"Streamlit"],
    'MLOps': ['Docker', 'MLflow', 'Git', 'CI/CD'],
    'Cloud': ['Azure', 'Google Cloud Platform']
}

PROJETS = [
        {
        'id': 1,
        'titre': 'High Dimensional Deep Learning Projects',
        'description': 'Regroupement de quatre mini-projets en deep learning appliqués à des données de haute dimension : CNN, apprentissage semi-supervisé, autoencodeur variationnel conditionnel (CVAE) et réseaux récurrents (RNN).',
        'technologies': ['Python', 'Jupyter Notebook', 'Deep Learning'],
        'github': 'https://github.com/mokhtar-khalil/High_dimensional_deep_learning_projects',
        'demo': '#',
        'image': 'hddl.png',
        'details': """Le projet comprend quatre notebooks distincts :
        - Projet CNN : réseau convolutif pour classification d’images haute dimension.
        - Projet SSL : apprentissage semi-supervisé pour tirer parti de données partiellement étiquetées.
        - Projet CVAE : génération d’échantillons via un autoencodeur variationnel conditionnel.
        - Projet RNN : traitement séquentiel avec des réseaux récurrents."""
    },
    {
    'id': 2,
    'titre': 'Movie Recommender System',
    'description': 'Système intelligent de recommandation et classification de films basé sur les affiches et les descriptions, intégrant CNN, MobileNet, ViT, TF-IDF, GloVe et BERT.',
    'technologies': ['Python', 'FastAPI', 'Gradio', 'Docker', 'TF-IDF', 'GloVe', 'BERT', 'MobileNet', 'ViT', 'Annoy'],
    'github': 'https://github.com/mokhtar-khalil/Movie_Recommender_System',
    'image': 'movie_recommender.png',
    'demo': '#',
    'details': """Ce projet comprend :
    - Classification des affiches de films par genre avec un modèle CNN.
    - Recommandation d’affiches similaires via MobileNet et ViT (extraction de caractéristiques et index Annoy).
    - Recommandation textuelle basée sur la description des films (TF-IDF, GloVe, BERT).
    - Interface utilisateur développée avec Gradio et API backend sous FastAPI.
    - Téléchargement automatique des modèles stockés sur Google Drive.
    - Conteneurisation complète via Docker et orchestration avec docker-compose."""
    },
    {
    'id': 3,
    'titre': 'Reinforcement Learning for Fluid Dynamics',
    'description': 'Application du Reinforcement Learning pour le contrôle d’un écoulement autour d’un cylindre en 2D.',
    'technologies': ['Python', 'FEniCS', 'NumPy', 'Matplotlib', 'Reinforcement Learning'],
    'github': 'https://github.com/mokhtar-khalil/Reinforcement_learning_fluids_dynamics',
    'image': 'rl_fluid_dynamics.png',
    'demo': '#',
    'details': """Le projet implémente un environnement de simulation d’écoulement 2D autour d’un cylindre 
    (Env2DCylinder) utilisant FEniCS pour la résolution numérique des équations de Navier–Stokes. 
    L’agent de Reinforcement Learning agit sur des jets de fluide pour contrôler le flux et réduire la traînée. 
    Les scripts incluent la génération de maillage, la définition des conditions aux limites et la boucle d’apprentissage 
    entre le solveur et l’agent. Le projet intègre des sorties de simulation et visualisations du maillage et des champs de vitesse."""
    }
    ,
    {
    'id': 4,
    'titre': 'Trading Simulation Agent',
    'description': 'Simulation de trading développée entièrement par des agents d’ingénierie autonomes utilisant l’architecture CrewAI.',
    'technologies': ['Python', 'Gradio', 'CrewAI', 'uv'],
    'github': 'https://github.com/mokhtar-khalil/Trading_simulation_Agent',
    'image': 'trading_simulation.jpg',
    'demo': '#',
    'details': """Le projet met en œuvre une simulation de trading conçue par une équipe d’agents IA autonomes :
    - Lead Engineer Agent : définit l’architecture et coordonne le travail.
    - Backend Engineer Agent : implémente la logique métier en Python.
    - Frontend Engineer Agent : crée une interface Gradio pour l’interaction.
    - Test Engineer Agent : écrit des tests unitaires pour assurer la qualité du code.
    L’application peut être exécutée depuis le dossier /example_output_4o via la commande `uv run app.py`.
    Le projet illustre la collaboration multi-agent pour la génération automatisée d’une application complète."""
    },
    {
    'id': 5,
    'titre': 'Velib Data Analysis',
    'description': 'Analyse des données de stations Vélib sur une semaine pour détecter des profils d’utilisation et prédire le taux de remplissage des stations.',
    'technologies': ['Python', 'R', 'Pandas', 'NumPy', 'Matplotlib', 'PCA', 'Clustering'],
    'github': 'https://github.com/mokhtar-khalil/Velib-data-analysis',
    'image': 'velib_analysis.jpg',
    'demo': '#',
    'details': """Le projet étudie les profils de chargement des stations Vélib sur la semaine du 2 au 7 septembre 2014.
    - Les individus sont les stations, et les variables représentent les 168 heures de la semaine.
    - Calcul du ratio de vélos disponibles par rapport au nombre total de bornes.
    - Application d’analyses statistiques descriptives et de la PCA pour la réduction de dimension.
    - Clustering des stations selon leurs profils temporels d’utilisation.
    - Objectif : identifier des groupes de comportements similaires et prédire le profil de charge d’une station.
    Le projet contient un notebook Python et un notebook R pour l’analyse complémentaire."""
}

]

EXPERIENCE = [
    {
        'poste': 'IA - Machine Learning Engineer',
        'entreprise': 'Groupe BPCE',
        'periode': 'Février 2025 - Août 2025',
        'description': [
            "Identification de cas d’usage IA à fort impact dans la gestion financière, en cohérence avec le plan stratégique 2030",
            "Développement d’un modèle d’optimisation des fenêtres de marché améliorant la précision du calendrier d’émission de la dette",
            "Sélection et déploiement du modèle le plus performant (XGBoost) avec intégration dans une interface Tkinter/FlaskAPI"
        ]
    },
    {
        'poste': 'Deep Reinforcement Learning Engineer',
        'entreprise': 'CNRS Toulouse',
        'periode': 'Juin 2024 - Septembre 2024',
        'description': [
            "Mise en relation du contrôle optimal avec le Reinforcement Learning via le principe de Bellman",
            "Réduction des forces de traînée et de portance dans un écoulement fluide en optimisant les équations de Navier-Stokes par PPO"
        ]
    },
    {
        'poste': 'Data Scientist',
        'entreprise': 'Data Turing',
        'periode': 'Mai 2023 - Juillet 2023',
        'description': [
            "Création d’une base d’apprentissage à partir d’offres de formation universitaires via du Data Scraping",
            "Utilisation de Flowise pour entraîner un chatbot sur les données collectées"
        ]
    }
]

EDUCATION = [
    {
        'diplome': 'Ingénieur en Mathématiques Appliquées',
        'ecole': 'Institut National des Sciences Appliquées de Toulouse (INSA)',
        'annee': '2025',
        'details': 'Spécialisation en Machine Learning et Big Data'
    },
    {
        'diplome': 'Ingénieur en Statistiques et Ingénierie des Données',
        'ecole': 'Ecole Supérieure Polytechnique de Nouakchott (ESP)',
        'annee': '2023',
        'details': 'Mention Très Bien - Major de promotion'
    }
]

CERTIFICATIONS = [
    'AI in AZURE Databricks',
    'Fondamentals of Computer Vision on AZURE',
    'Fondamentals of Facial Recognition on AZURE',
    'Google Cloud Big Data and Machine Learning Fundamentals'
]

@app.route('/')
def accueil():
    return render_template('index.html', 
                         profil=PROFIL, 
                         projets=PROJETS[:3],
                         competences=COMPETENCES)

@app.route('/projets')
def projets():
    return render_template('projets.html', 
                         profil=PROFIL,
                         projets=PROJETS)

@app.route('/projet/<int:projet_id>')
def projet_detail(projet_id):
    projet = next((p for p in PROJETS if p['id'] == projet_id), None)
    if projet:
        return render_template('projet_detail.html', 
                             profil=PROFIL,
                             projet=projet)
    return "Projet non trouvé", 404

@app.route('/a-propos')
def a_propos():
    return render_template('apropos.html', 
                         profil=PROFIL,
                         experience=EXPERIENCE,
                         education=EDUCATION,
                         certifications=CERTIFICATIONS,
                         competences=COMPETENCES)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        data = request.form
        # Ici vous pouvez ajouter l'envoi d'email ou sauvegarde en base
        return jsonify({'status': 'success', 'message': 'Message envoyé avec succès!'})
    return render_template('contact.html', profil=PROFIL)

@app.route('/cv')
def cv():
    return render_template('cv.html',
                         profil=PROFIL,
                         experience=EXPERIENCE,
                         education=EDUCATION,
                         certifications=CERTIFICATIONS,
                         competences=COMPETENCES)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)