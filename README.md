# ⚽ Football Analytics Pro

Application Streamlit pour l'analyse statistique avancée des joueurs de football FBref 2024-2025, avec intégration Ollama (LLM local).

## 🚀 Installation & Lancement

### 1. Installer les dépendances Python
```bash
pip install -r requirements.txt
```

### 2. Installer & démarrer Ollama
```bash
# Linux / macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Télécharger un modèle
ollama pull llama3.2
# ou
ollama pull mistral
# ou  
ollama pull qwen2.5:7b

# Démarrer le serveur
ollama serve
```

### 3. Lancer l'application
```bash
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`

---

## 📊 Fonctionnalités

### Onglets analytiques
| Onglet | Contenu |
|--------|---------|
| 📈 Vue Globale | Buts vs xG, Top 15 joueurs, moyennes mobiles |
| 📉 Stationnarité | Tests ADF + KPSS avec interprétation |
| 🔄 Saisonnalité | Décomposition Tendance / Saisonnalité / Résidus |
| 📐 ACF / PACF | Autocorrélation & autocorrélation partielle |
| 🧮 Distribution | Histogramme, KDE, Q-Q plot, Shapiro-Wilk |
| 🔗 Corrélations | Heatmap + top paires corrélées |
| 🤖 Chat IA | Interface conversationnelle Ollama |

### Filtres sidebar
- Position (FW, AT, MF, DF, GK)
- Matchs minimum joués
- Tranche d'âge
- Métrique principale
- Nombre de lags ACF/PACF

---

## 🏗️ Structure du projet
```
project/
├── app.py              ← Application principale
├── requirements.txt    ← Dépendances
├── README.md          ← Ce fichier
├── data/              ← Données locales (optionnel)
├── modules/           ← Modules analytiques
└── scripts/           ← Scripts de preprocessing
```

## ⚙️ Configuration Ollama
- **Host par défaut** : `http://localhost:11434`
- Changeable dans la sidebar
- Modèles recommandés : `llama3.2`, `mistral`, `qwen2.5`

---

## 📈 Interprétation des tests statistiques

### Stationnarité (ADF + KPSS)
| ADF | KPSS | Verdict |
|-----|------|---------|
| ✅ p<0.05 | ✅ p>0.05 | Stationnaire → modélisable directement |
| ❌ p>0.05 | ❌ p<0.05 | Non-stationnaire → différencier |
| Contradictoire | | Tendance stochastique vs déterministe |

### ACF / PACF
- Barres dépassant les tirets jaunes = lags significatifs
- ACF décroît lentement → série non-stationnaire
- PACF coupe après lag k → modèle AR(k)
