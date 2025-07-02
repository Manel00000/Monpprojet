{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "9ed11c2a-2873-40db-a8e7-da096d7d0521",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "C:\\Users\\manel\\anaconda3\\envs\\myenv\\python.exe\n"
     ]
    }
   ],
   "source": [
    "import sys\n",
    "print(sys.executable)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2a0abf10-2e03-473b-93d9-52e2dd26523b",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'xgboost'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 7\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mjoblib\u001b[39;00m\n\u001b[0;32m      6\u001b[0m \u001b[38;5;66;03m# Charger les modèles\u001b[39;00m\n\u001b[1;32m----> 7\u001b[0m nb_sinistre_model \u001b[38;5;241m=\u001b[39m \u001b[43mjoblib\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mload\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mnd_sinistre_xgb.joblib\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m)\u001b[49m\n\u001b[0;32m      8\u001b[0m cout_model \u001b[38;5;241m=\u001b[39m joblib\u001b[38;5;241m.\u001b[39mload(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcout_xgb.joblib\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m     10\u001b[0m st\u001b[38;5;241m.\u001b[39mtitle(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mCalcul des primes\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "File \u001b[1;32m~\\anaconda3\\envs\\myenv\\lib\\site-packages\\joblib\\numpy_pickle.py:749\u001b[0m, in \u001b[0;36mload\u001b[1;34m(filename, mmap_mode, ensure_native_byte_order)\u001b[0m\n\u001b[0;32m    744\u001b[0m                 \u001b[38;5;28;01mreturn\u001b[39;00m load_compatibility(fobj)\n\u001b[0;32m    746\u001b[0m             \u001b[38;5;66;03m# A memory-mapped array has to be mapped with the endianness\u001b[39;00m\n\u001b[0;32m    747\u001b[0m             \u001b[38;5;66;03m# it has been written with. Other arrays are coerced to the\u001b[39;00m\n\u001b[0;32m    748\u001b[0m             \u001b[38;5;66;03m# native endianness of the host system.\u001b[39;00m\n\u001b[1;32m--> 749\u001b[0m             obj \u001b[38;5;241m=\u001b[39m \u001b[43m_unpickle\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    750\u001b[0m \u001b[43m                \u001b[49m\u001b[43mfobj\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    751\u001b[0m \u001b[43m                \u001b[49m\u001b[43mensure_native_byte_order\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mensure_native_byte_order\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    752\u001b[0m \u001b[43m                \u001b[49m\u001b[43mfilename\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mfilename\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    753\u001b[0m \u001b[43m                \u001b[49m\u001b[43mmmap_mode\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mvalidated_mmap_mode\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    754\u001b[0m \u001b[43m            \u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    756\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m obj\n",
      "File \u001b[1;32m~\\anaconda3\\envs\\myenv\\lib\\site-packages\\joblib\\numpy_pickle.py:626\u001b[0m, in \u001b[0;36m_unpickle\u001b[1;34m(fobj, ensure_native_byte_order, filename, mmap_mode)\u001b[0m\n\u001b[0;32m    624\u001b[0m obj \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m    625\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m--> 626\u001b[0m     obj \u001b[38;5;241m=\u001b[39m \u001b[43munpickler\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mload\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    627\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m unpickler\u001b[38;5;241m.\u001b[39mcompat_mode:\n\u001b[0;32m    628\u001b[0m         warnings\u001b[38;5;241m.\u001b[39mwarn(\n\u001b[0;32m    629\u001b[0m             \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mThe file \u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;132;01m%s\u001b[39;00m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m has been generated with a \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    630\u001b[0m             \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mjoblib version less than 0.10. \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    633\u001b[0m             stacklevel\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m3\u001b[39m,\n\u001b[0;32m    634\u001b[0m         )\n",
      "File \u001b[1;32m~\\anaconda3\\envs\\myenv\\lib\\pickle.py:1213\u001b[0m, in \u001b[0;36m_Unpickler.load\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m   1211\u001b[0m             \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mEOFError\u001b[39;00m\n\u001b[0;32m   1212\u001b[0m         \u001b[38;5;28;01massert\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(key, bytes_types)\n\u001b[1;32m-> 1213\u001b[0m         \u001b[43mdispatch\u001b[49m\u001b[43m[\u001b[49m\u001b[43mkey\u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;241;43m0\u001b[39;49m\u001b[43m]\u001b[49m\u001b[43m]\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[43m)\u001b[49m\n\u001b[0;32m   1214\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m _Stop \u001b[38;5;28;01mas\u001b[39;00m stopinst:\n\u001b[0;32m   1215\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m stopinst\u001b[38;5;241m.\u001b[39mvalue\n",
      "File \u001b[1;32m~\\anaconda3\\envs\\myenv\\lib\\pickle.py:1538\u001b[0m, in \u001b[0;36m_Unpickler.load_stack_global\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m   1536\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mtype\u001b[39m(name) \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28mstr\u001b[39m \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28mtype\u001b[39m(module) \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28mstr\u001b[39m:\n\u001b[0;32m   1537\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m UnpicklingError(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mSTACK_GLOBAL requires str\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m-> 1538\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mappend(\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mfind_class\u001b[49m\u001b[43m(\u001b[49m\u001b[43mmodule\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mname\u001b[49m\u001b[43m)\u001b[49m)\n",
      "File \u001b[1;32m~\\anaconda3\\envs\\myenv\\lib\\pickle.py:1580\u001b[0m, in \u001b[0;36m_Unpickler.find_class\u001b[1;34m(self, module, name)\u001b[0m\n\u001b[0;32m   1578\u001b[0m     \u001b[38;5;28;01melif\u001b[39;00m module \u001b[38;5;129;01min\u001b[39;00m _compat_pickle\u001b[38;5;241m.\u001b[39mIMPORT_MAPPING:\n\u001b[0;32m   1579\u001b[0m         module \u001b[38;5;241m=\u001b[39m _compat_pickle\u001b[38;5;241m.\u001b[39mIMPORT_MAPPING[module]\n\u001b[1;32m-> 1580\u001b[0m \u001b[38;5;28;43m__import__\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mmodule\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mlevel\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;241;43m0\u001b[39;49m\u001b[43m)\u001b[49m\n\u001b[0;32m   1581\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mproto \u001b[38;5;241m>\u001b[39m\u001b[38;5;241m=\u001b[39m \u001b[38;5;241m4\u001b[39m:\n\u001b[0;32m   1582\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m _getattribute(sys\u001b[38;5;241m.\u001b[39mmodules[module], name)[\u001b[38;5;241m0\u001b[39m]\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'xgboost'"
     ]
    }
   ],
   "source": [
    "\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "\n",
    "# Charger les modèles\n",
    "nb_sinistre_model = joblib.load(\"nd_sinistre_xgb.joblib\")\n",
    "cout_model = joblib.load(\"cout_xgb.joblib\")\n",
    "\n",
    "st.title(\"Calcul des primes\")\n",
    "\n",
    "# Variables catégorielles (avec virgules dans les listes)\n",
    "type_vehicule = st.selectbox(\"Type de véhicule\", [\n",
    "    'USAGE PRIVE', 'UTILITAIRE I', 'UTILITAIRE II', 'ENGINE AGRICOLE',\n",
    "    'AUTO ECOLE', 'LOCATION', 'LOUAGE', 'TAXI', '2 ROUES', 'QUAD'\n",
    "])\n",
    "region = st.selectbox(\"Zone géographique\", [\n",
    "    'Grand Tunis', 'sfax', 'sahel', 'nord', 'sud', 'JERBA', 'nabeul'\n",
    "])\n",
    "\n",
    "# Variables numériques\n",
    "age_vehicule = st.number_input(\"Âge du véhicule (en mois)\", min_value=0, max_value=300, value=60)\n",
    "\n",
    "# Construire un dataframe avec les entrées utilisateur\n",
    "data = pd.DataFrame({\n",
    "    \"type_vehicule\": [type_vehicule],  # correction ici\n",
    "    \"region\": [region],\n",
    "    \"age véhicule\": [age_vehicule]\n",
    "})\n",
    "\n",
    "# Préprocessing\n",
    "data[\"age véhicule\"] = data[\"age véhicule\"] / 12  # convertir en années\n",
    "\n",
    "# One-hot encoding des variables catégorielles\n",
    "data_encoded = pd.get_dummies(data, columns=[\"type_vehicule\", \"region\"], drop_first=True)\n",
    "\n",
    "# Colonnes attendues par le modèle (à adapter si besoin)\n",
    "expected_columns = nb_sinistre_model.get_booster().feature_names\n",
    "\n",
    "# Ajouter colonnes manquantes avec 0\n",
    "for col in expected_columns:\n",
    "    if col not in data_encoded.columns:\n",
    "        data_encoded[col] = 0\n",
    "\n",
    "# Réordonner les colonnes pour correspondre à l’ordre du modèle\n",
    "data_encoded = data_encoded[expected_columns]\n",
    "\n",
    "# Bouton pour lancer la prédiction\n",
    "if st.button(\"Calculer la prime\"):\n",
    "    # Prédictions\n",
    "    nb_pred = nb_sinistre_model.predict(data_encoded)[0]\n",
    "    cout_log_pred = cout_model.predict(data_encoded)[0]\n",
    "\n",
    "    # Inverse transformation du coût\n",
    "    cout_pred = np.expm1(cout_log_pred)\n",
    "\n",
    "    # Calcul de la prime\n",
    "    prime = nb_pred * cout_pred\n",
    "\n",
    "    st.write(f\"Nombre de sinistres prédit : {nb_pred:.3f}\")\n",
    "    st.write(f\"Coût sinistre prédit : {cout_pred:.2f}\")\n",
    "    st.write(f\"Prime calculée : {prime:.2f}\")\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (myenv)",
   "language": "python",
   "name": "myenv"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
