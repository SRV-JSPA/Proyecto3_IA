import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import re
import warnings
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import csv
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

@dataclass
class Mensaje:
    texto: str
    etiqueta: str
    texto_preprocesado: str = ""
    longitud: int = 0
    longitud_preprocesado: int = 0
    
    def __post_init__(self):
        self.longitud = len(self.texto) if self.texto else 0
        self.longitud_preprocesado = len(self.texto_preprocesado) if self.texto_preprocesado else 0


@dataclass
class ResultadoAnalisis:
    distribucion: Dict[str, int] = field(default_factory=dict)
    proporcion: Dict[str, float] = field(default_factory=dict)
    longitud_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    palabras_frecuentes_ham: List[Tuple[str, int]] = field(default_factory=list)
    palabras_frecuentes_spam: List[Tuple[str, int]] = field(default_factory=list)
    palabras_frecuentes_ham_prep: List[Tuple[str, int]] = field(default_factory=list)
    palabras_frecuentes_spam_prep: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class VocabularioBayes:
    palabras_spam: Dict[str, int] = field(default_factory=dict)
    palabras_ham: Dict[str, int] = field(default_factory=dict)
    prob_palabra_spam: Dict[str, float] = field(default_factory=dict)
    prob_palabra_ham: Dict[str, float] = field(default_factory=dict)
    prob_spam: float = 0.0
    prob_ham: float = 0.0
    total_palabras_spam: int = 0
    total_palabras_ham: int = 0
    palabras_mas_predictivas: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class MetricasEvaluacion:
    matriz_confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    resultados_thresholds: Dict[float, Dict[str, float]] = field(default_factory=dict)
    mejor_threshold: float = 0.5


class Visualizador:
    def __init__(self, guardar_figuras=True, directorio="figuras_spam_ham"):
        self.guardar_figuras = guardar_figuras
        self.directorio = directorio
        
        plt.ion()  
        
        if self.guardar_figuras and not os.path.exists(self.directorio):
            os.makedirs(self.directorio)
    
    def visualizar_o_guardar(self, fig, nombre_archivo):
        if self.guardar_figuras:
            ruta_completa = os.path.join(self.directorio, f"{nombre_archivo}.png")
            fig.savefig(ruta_completa, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"Figura guardada como: {ruta_completa}")
        else:
            plt.show(block=False)
            plt.pause(0.5)
    
    def visualizar_distribucion(self, resultado: ResultadoAnalisis):
        fig = plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(resultado.distribucion.keys(), resultado.distribucion.values())
        plt.title('Conteo de mensajes por tipo')
        plt.xlabel('Tipo de mensaje')
        plt.ylabel('Cantidad')
        
        plt.subplot(1, 2, 2)
        plt.pie(resultado.distribucion.values(), labels=resultado.distribucion.keys(), 
                autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Proporción de mensajes')
        
        plt.tight_layout()
        self.visualizar_o_guardar(fig, "1_distribucion_mensajes")
    
    def visualizar_longitud(self, mensajes: List[Mensaje], resultado: ResultadoAnalisis):
        df_temp = pd.DataFrame({
            'label': [m.etiqueta for m in mensajes],
            'longitud': [m.longitud for m in mensajes]
        })
        
        fig = plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.kdeplot(data=df_temp[df_temp['label']=='ham'], x='longitud', label='Ham', fill=True, alpha=0.5)
        sns.kdeplot(data=df_temp[df_temp['label']=='spam'], x='longitud', label='Spam', fill=True, alpha=0.5)
        plt.title('Densidad de longitud por tipo de mensaje')
        plt.xlabel('Longitud del mensaje')
        plt.ylabel('Densidad')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x='label', y='longitud', data=df_temp)
        plt.title('Distribución de longitud por tipo de mensaje')
        plt.xlabel('Tipo de mensaje')
        plt.ylabel('Longitud')
        
        plt.tight_layout()
        self.visualizar_o_guardar(fig, "2_longitud_mensajes")
    
    def visualizar_palabras_frecuentes(self, resultado: ResultadoAnalisis, preprocesado=False):
        if preprocesado:
            ham_words = resultado.palabras_frecuentes_ham_prep
            spam_words = resultado.palabras_frecuentes_spam_prep
            nombre_archivo = "6_palabras_frecuentes_preprocesado"
        else:
            ham_words = resultado.palabras_frecuentes_ham
            spam_words = resultado.palabras_frecuentes_spam
            nombre_archivo = "3_palabras_frecuentes"
        
        if not ham_words or not spam_words:
            print("No hay suficientes palabras para visualizar")
            return
        
        fig = plt.figure(figsize=(14, 8))
        
        plt.subplot(1, 2, 1)
        words, counts = zip(*ham_words)
        plt.barh(list(reversed(words)), list(reversed(counts)))
        plt.title(f'Top {len(ham_words)} palabras más frecuentes en Ham')
        plt.xlabel('Frecuencia')
        
        plt.subplot(1, 2, 2)
        words, counts = zip(*spam_words)
        plt.barh(list(reversed(words)), list(reversed(counts)))
        plt.title(f'Top {len(spam_words)} palabras más frecuentes en Spam')
        plt.xlabel('Frecuencia')
        
        plt.tight_layout()
        self.visualizar_o_guardar(fig, nombre_archivo)
    
    def visualizar_wordcloud(self, mensajes: List[Mensaje], preprocesado=False):
        if preprocesado:
            ham_text = ' '.join([m.texto_preprocesado for m in mensajes if m.etiqueta == 'ham'])
            spam_text = ' '.join([m.texto_preprocesado for m in mensajes if m.etiqueta == 'spam'])
            nombre_archivo = "7_wordcloud_preprocesado"
        else:
            ham_text = ' '.join([m.texto for m in mensajes if m.etiqueta == 'ham'])
            spam_text = ' '.join([m.texto for m in mensajes if m.etiqueta == 'spam'])
            nombre_archivo = "4_wordcloud"
        
        wc = WordCloud(width=800, height=400, background_color='white', max_words=200, contour_width=3)
        
        fig = plt.figure(figsize=(14, 7))
        
        plt.subplot(1, 2, 1)
        ham_wc = wc.generate(ham_text)
        plt.imshow(ham_wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('WordCloud - Ham', fontsize=15)
        
        plt.subplot(1, 2, 2)
        spam_wc = wc.generate(spam_text)
        plt.imshow(spam_wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('WordCloud - Spam', fontsize=15)
        
        plt.tight_layout()
        self.visualizar_o_guardar(fig, nombre_archivo)
    
    def visualizar_longitud_preprocesado(self, mensajes: List[Mensaje]):
        df_temp = pd.DataFrame({
            'label': [m.etiqueta for m in mensajes],
            'longitud': [m.longitud for m in mensajes],
            'longitud_preprocesado': [m.longitud_preprocesado for m in mensajes]
        })
        
        fig = plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.kdeplot(data=df_temp[df_temp['label']=='ham'], x='longitud_preprocesado', 
                   label='Ham', fill=True, alpha=0.5)
        sns.kdeplot(data=df_temp[df_temp['label']=='spam'], x='longitud_preprocesado', 
                   label='Spam', fill=True, alpha=0.5)
        plt.title('Densidad de longitud por tipo (preprocesado)')
        plt.xlabel('Longitud del mensaje preprocesado')
        plt.ylabel('Densidad')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(df_temp['longitud'], df_temp['longitud_preprocesado'], alpha=0.5, 
                   c=df_temp['label'].map({'ham': 'blue', 'spam': 'red'}))
        plt.title('Longitud original vs preprocesada')
        plt.xlabel('Longitud original')
        plt.ylabel('Longitud preprocesada')
        
        plt.tight_layout()
        self.visualizar_o_guardar(fig, "5_longitud_preprocesada")
    
    def visualizar_matriz_confusion(self, metricas: MetricasEvaluacion):
        if not metricas.matriz_confusion:
            print("No hay datos de matriz de confusión disponibles")
            return
        
        fig = plt.figure(figsize=(8, 6))
        
        matriz = np.array([
            [metricas.matriz_confusion.get('ham', {}).get('ham', 0),
             metricas.matriz_confusion.get('ham', {}).get('spam', 0)],
            [metricas.matriz_confusion.get('spam', {}).get('ham', 0),
             metricas.matriz_confusion.get('spam', {}).get('spam', 0)]
        ])
        
        sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam'])
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        
        plt.tight_layout()
        self.visualizar_o_guardar(fig, "8_matriz_confusion")
    
    def visualizar_thresholds(self, metricas: MetricasEvaluacion):
        if not metricas.resultados_thresholds:
            print("No hay datos de thresholds disponibles")
            return
        
        thresholds = sorted(metricas.resultados_thresholds.keys())
        precision = [metricas.resultados_thresholds[t]['precision'] for t in thresholds]
        recall = [metricas.resultados_thresholds[t]['recall'] for t in thresholds]
        f1 = [metricas.resultados_thresholds[t]['f1'] for t in thresholds]
        
        fig = plt.figure(figsize=(10, 6))
        
        plt.plot(thresholds, precision, label='Precision', marker='o')
        plt.plot(thresholds, recall, label='Recall', marker='s')
        plt.plot(thresholds, f1, label='F1-Score', marker='^')
        plt.axvline(x=metricas.mejor_threshold, color='r', linestyle='--', 
                   label=f'Mejor Threshold ({metricas.mejor_threshold})')
        
        plt.title('Métricas de Rendimiento por Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.visualizar_o_guardar(fig, "9_thresholds")
    
    def visualizar_palabras_predictivas(self, vocabulario: VocabularioBayes):
        if not vocabulario.palabras_mas_predictivas:
            print("No hay datos de palabras predictivas disponibles")
            return
        
        top_palabras = vocabulario.palabras_mas_predictivas[:10]
        words, scores = zip(*top_palabras)
        
        fig = plt.figure(figsize=(10, 6))
        
        plt.barh(list(reversed(words)), list(reversed(scores)))
        plt.title('Top 10 Palabras más Predictivas para SPAM')
        plt.xlabel('Probabilidad P(S|W)')
        
        plt.tight_layout()
        self.visualizar_o_guardar(fig, "10_palabras_predictivas")


class ClasificadorBayesiano:
    def __init__(self, laplace_smoothing=1):
        self.vocabulario = VocabularioBayes()
        self.laplace_smoothing = laplace_smoothing
        self.metricas = MetricasEvaluacion()
    
    def entrenar(self, mensajes_train: List[Mensaje]):
        total_mensajes = len(mensajes_train)
        mensajes_spam = [m for m in mensajes_train if m.etiqueta == 'spam']
        mensajes_ham = [m for m in mensajes_train if m.etiqueta == 'ham']
        
        self.vocabulario.prob_spam = len(mensajes_spam) / total_mensajes
        self.vocabulario.prob_ham = len(mensajes_ham) / total_mensajes
        
        print(f"Probabilidad a priori de SPAM: {self.vocabulario.prob_spam:.4f}")
        print(f"Probabilidad a priori de HAM: {self.vocabulario.prob_ham:.4f}")
        
        for mensaje in mensajes_train:
            if not mensaje.texto_preprocesado:
                continue
                
            palabras = mensaje.texto_preprocesado.split()
            
            palabras_unicas = set(palabras)
            
            for palabra in palabras_unicas:
                if mensaje.etiqueta == 'spam':
                    self.vocabulario.palabras_spam[palabra] = self.vocabulario.palabras_spam.get(palabra, 0) + 1
                else:
                    self.vocabulario.palabras_ham[palabra] = self.vocabulario.palabras_ham.get(palabra, 0) + 1
        
        self.vocabulario.total_palabras_spam = sum(self.vocabulario.palabras_spam.values())
        self.vocabulario.total_palabras_ham = sum(self.vocabulario.palabras_ham.values())
        
        vocabulario_completo = set(list(self.vocabulario.palabras_spam.keys()) + 
                                  list(self.vocabulario.palabras_ham.keys()))
        
        for palabra in vocabulario_completo:
            self.vocabulario.prob_palabra_spam[palabra] = (
                self.vocabulario.palabras_spam.get(palabra, 0) + self.laplace_smoothing
            ) / (self.vocabulario.total_palabras_spam + self.laplace_smoothing * len(vocabulario_completo))
            
            self.vocabulario.prob_palabra_ham[palabra] = (
                self.vocabulario.palabras_ham.get(palabra, 0) + self.laplace_smoothing
            ) / (self.vocabulario.total_palabras_ham + self.laplace_smoothing * len(vocabulario_completo))
        
        palabras_predictivas = []
        
        for palabra in vocabulario_completo:
            p_w_s = self.vocabulario.prob_palabra_spam[palabra]
            p_w_h = self.vocabulario.prob_palabra_ham[palabra]
            p_s = self.vocabulario.prob_spam
            
            p_s_dado_w = (p_w_s * p_s) / (p_w_s * p_s + p_w_h * (1 - p_s))
            
            palabras_predictivas.append((palabra, p_s_dado_w))
        
        palabras_predictivas.sort(key=lambda x: x[1], reverse=True)
        self.vocabulario.palabras_mas_predictivas = palabras_predictivas
        
        print(f"\nTop 10 palabras más predictivas para SPAM:")
        for palabra, prob in palabras_predictivas[:10]:
            print(f"  {palabra}: {prob:.4f}")
    
    def predecir_mensaje(self, texto_preprocesado: str, threshold=0.5) -> Tuple[str, float, List[Tuple[str, float]]]:
        if not texto_preprocesado:
            return "ham", 0.0, []
        
        palabras = texto_preprocesado.split()
        
        palabras_conocidas = [p for p in palabras if p in self.vocabulario.prob_palabra_spam]
        
        if not palabras_conocidas:
            return "spam" if self.vocabulario.prob_spam > threshold else "ham", self.vocabulario.prob_spam, []
        
        probs_individuales = []
        for palabra in palabras_conocidas:
            p_w_s = self.vocabulario.prob_palabra_spam[palabra]
            p_w_h = self.vocabulario.prob_palabra_ham[palabra]
            p_s = self.vocabulario.prob_spam
            p_h = self.vocabulario.prob_ham
            
            p_s_dado_w = (p_w_s * p_s) / (p_w_s * p_s + p_w_h * p_h)
            
            probs_individuales.append((palabra, p_s_dado_w))
        
        probs_individuales.sort(key=lambda x: x[1], reverse=True)
        

        p_productos = 1.0
        p_complementos = 1.0
        
        for _, p_s_w in probs_individuales:
            p_productos *= p_s_w
            p_complementos *= (1 - p_s_w)
        
        if p_productos + p_complementos == 0:
            prob_spam = self.vocabulario.prob_spam
        else:
            prob_spam = p_productos / (p_productos + p_complementos)
        
        etiqueta = "spam" if prob_spam > threshold else "ham"
        
        top_palabras = probs_individuales[:3] if len(probs_individuales) >= 3 else probs_individuales
        
        return etiqueta, prob_spam, top_palabras
    
    def evaluar(self, mensajes_test: List[Mensaje], threshold=0.5) -> MetricasEvaluacion:
        matriz_confusion = {
            'ham': {'ham': 0, 'spam': 0},
            'spam': {'ham': 0, 'spam': 0}
        }
        
        y_true = []
        y_pred = []
        
        for mensaje in mensajes_test:
            etiqueta_real = mensaje.etiqueta
            etiqueta_pred, _, _ = self.predecir_mensaje(mensaje.texto_preprocesado, threshold)
            
            matriz_confusion[etiqueta_real][etiqueta_pred] += 1
            
            y_true.append(1 if etiqueta_real == 'spam' else 0)
            y_pred.append(1 if etiqueta_pred == 'spam' else 0)
        
        precision = precision_score(y_true, y_pred) if sum(y_pred) > 0 else 0
        recall = recall_score(y_true, y_pred) if sum(y_true) > 0 else 0
        f1 = f1_score(y_true, y_pred) if (precision + recall) > 0 else 0
        
        metricas = MetricasEvaluacion(
            matriz_confusion=matriz_confusion,
            precision=precision,
            recall=recall,
            f1_score=f1
        )
        
        return metricas
    
    def optimizar_threshold(self, mensajes_test: List[Mensaje], thresholds=None):
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        resultados = {}
        mejor_f1 = -1
        mejor_threshold = 0.5
        
        for threshold in thresholds:
            metricas = self.evaluar(mensajes_test, threshold)
            
            resultados[threshold] = {
                'precision': metricas.precision,
                'recall': metricas.recall,
                'f1': metricas.f1_score
            }
            
            print(f"Threshold: {threshold:.1f}, Precision: {metricas.precision:.4f}, "
                  f"Recall: {metricas.recall:.4f}, F1: {metricas.f1_score:.4f}")
            
            if metricas.f1_score > mejor_f1:
                mejor_f1 = metricas.f1_score
                mejor_threshold = threshold
        
        print(f"\nMejor threshold: {mejor_threshold:.1f} con F1-Score: {mejor_f1:.4f}")
        
        self.metricas = self.evaluar(mensajes_test, mejor_threshold)
        self.metricas.resultados_thresholds = resultados
        self.metricas.mejor_threshold = mejor_threshold
        
        return self.metricas


class AnalizadorSpamHam:
    def __init__(self, test_size=0.2, random_seed=42):
        self.mensajes = []
        self.mensajes_train = []
        self.mensajes_test = []
        self.resultado = ResultadoAnalisis()
        self.visualizador = Visualizador()
        self.clasificador = ClasificadorBayesiano()
        self.test_size = test_size
        self.random_seed = random_seed
        
        print("Descargando recursos NLTK necesarios...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
    
    def cargar_datos(self, ruta_archivo: str) -> None:
        try:
            df = pd.read_csv(ruta_archivo, encoding='cp1252', sep=';', quoting=1, 
                           quotechar='"', on_bad_lines='warn')  
        except Exception as e:
            print(f"Error con el método 1: {e}")
            try:
                messages = []
                labels = []
                
                with open(ruta_archivo, 'r', encoding='cp1252') as f:
                    reader = csv.reader(f, delimiter=';')
                    next(reader)
                    for row in reader:
                        if len(row) >= 2:  
                            labels.append(row[0])
                            messages.append(row[1])
                
                df = pd.DataFrame({'label': labels, 'text': messages})
            except Exception as e2:
                print(f"Error con el método 2: {e2}")
                df = pd.read_csv(ruta_archivo, encoding='cp1252', sep=';', engine='python')

        if 'Label' in df.columns and 'SMS_TEXT' in df.columns:
            df.columns = ['label', 'text']

        df['label'] = df['label'].str.strip().str.lower()
        df['label'] = df['label'].replace(['ham"', 'ham"""', 'ham""\\"'], 'ham')

        df['text'] = df['text'].fillna('').astype(str)
        
        self.mensajes = [Mensaje(texto=texto, etiqueta=etiqueta) 
                        for texto, etiqueta in zip(df['text'], df['label'])]
        
        print("Dimensiones del dataset:", len(self.mensajes))
        print("\nPrimeras 5 filas del dataset:")
        for i in range(min(5, len(self.mensajes))):
            print(f"{self.mensajes[i].etiqueta}: {self.mensajes[i].texto[:50]}...")
        
        etiquetas = [m.etiqueta for m in self.mensajes]
        self.resultado.distribucion = dict(Counter(etiquetas))
        total = len(etiquetas)
        self.resultado.proporcion = {k: round(v/total*100, 2) for k, v in self.resultado.distribucion.items()}
        
        print("\nDistribución de etiquetas después de limpieza:")
        for etiqueta, cantidad in self.resultado.distribucion.items():
            print(f"{etiqueta}: {cantidad}")
    
    def mostrar_mensajes_aleatorios(self, n=5) -> None:
        if not self.mensajes:
            print("No hay mensajes cargados")
            return
        
        indices = random.sample(range(len(self.mensajes)), k=min(n, len(self.mensajes)))
        print("\nMensajes aleatorios:")
        for i in indices:
            print(f"{self.mensajes[i].etiqueta}: {self.mensajes[i].texto[:50]}...")
    
    def analizar_distribucion(self) -> None:
        if not self.mensajes:
            print("No hay mensajes cargados")
            return
        
        print("\nDistribución de mensajes:")
        print(f"Conteo:")
        for etiqueta, cantidad in self.resultado.distribucion.items():
            print(f"{etiqueta}: {cantidad}")
        
        print(f"Proporción (%):")
        for etiqueta, proporcion in self.resultado.proporcion.items():
            print(f"{etiqueta}: {proporcion}%")
        
        self.visualizador.visualizar_distribucion(self.resultado)
    
    def analizar_longitud_mensajes(self) -> None:
        if not self.mensajes:
            print("No hay mensajes cargados")
            return
        
        self.resultado.longitud_stats = {}
        for etiqueta in self.resultado.distribucion.keys():
            longitudes = [m.longitud for m in self.mensajes if m.etiqueta == etiqueta]
            self.resultado.longitud_stats[etiqueta] = {
                'count': len(longitudes),
                'mean': sum(longitudes) / len(longitudes) if longitudes else 0,
                'std': (sum((x - (sum(longitudes) / len(longitudes)))**2 for x in longitudes) / len(longitudes))**0.5 if longitudes else 0,
                'min': min(longitudes) if longitudes else 0,
                'max': max(longitudes) if longitudes else 0
            }
        
        print("\nLongitud de los mensajes:")
        for etiqueta, stats in self.resultado.longitud_stats.items():
            print(f"{etiqueta}: {stats}")
        
        self.visualizador.visualizar_longitud(self.mensajes, self.resultado)
    
    def analizar_palabras_frecuentes(self, n=20) -> None:
        if not self.mensajes:
            print("No hay mensajes cargados")
            return
        
        def get_words(text):
            if not text or not isinstance(text, str):
                return []
            return text.lower().split()
        
        ham_words = []
        spam_words = []
        
        for mensaje in self.mensajes:
            if mensaje.etiqueta == 'ham':
                ham_words.extend(get_words(mensaje.texto))
            elif mensaje.etiqueta == 'spam':
                spam_words.extend(get_words(mensaje.texto))
        
        ham_word_counts = Counter(ham_words)
        spam_word_counts = Counter(spam_words)
        
        self.resultado.palabras_frecuentes_ham = ham_word_counts.most_common(n)
        self.resultado.palabras_frecuentes_spam = spam_word_counts.most_common(n)
        
        print("\nAnálisis de palabras frecuentes:")
        
        self.visualizador.visualizar_palabras_frecuentes(self.resultado)
    
    def generar_wordcloud(self) -> None:
        if not self.mensajes:
            print("No hay mensajes cargados")
            return
        
        print("\nVisualización con WordCloud:")
        self.visualizador.visualizar_wordcloud(self.mensajes)
    
    def tokenizar_manual(self, texto):
        if not isinstance(texto, str):
            texto = str(texto)
        
        return re.findall(r'\b\w+\b', texto.lower())
    
    def preprocesar_texto(self, texto):
        if not texto or not isinstance(texto, str):
            return ""
        
        try:
            texto = texto.lower()
            
            texto = re.sub(r'[^a-zA-Z\s]', '', texto)
            
            try:
                tokens = word_tokenize(texto)
            except:
                tokens = self.tokenizar_manual(texto)
            
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
            
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(token) for token in tokens]
            
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error al preprocesar texto: {e}")
            return re.sub(r'[^\w\s]', '', texto.lower())
    
    def aplicar_preprocesamiento(self) -> None:
        if not self.mensajes:
            print("No hay mensajes cargados")
            return
        
        print("\nPreprocesamiento de texto:")
        for mensaje in self.mensajes:
            mensaje.texto_preprocesado = self.preprocesar_texto(mensaje.texto)
            mensaje.longitud_preprocesado = len(mensaje.texto_preprocesado)
        
        print("Ejemplos de texto preprocesado:")
        for i in range(min(5, len(self.mensajes))):
            print(f"Original: {self.mensajes[i].texto[:50]}...")
            print(f"Preprocesado: {self.mensajes[i].texto_preprocesado[:50]}...")
    
    def analizar_longitud_preprocesado(self) -> None:
        if not self.mensajes:
            print("No hay mensajes cargados")
            return
        
        longitud_prep_stats = {}
        for etiqueta in self.resultado.distribucion.keys():
            longitudes = [m.longitud_preprocesado for m in self.mensajes if m.etiqueta == etiqueta]
            longitud_prep_stats[etiqueta] = {
                'count': len(longitudes),
                'mean': sum(longitudes) / len(longitudes) if longitudes else 0,
                'std': (sum((x - (sum(longitudes) / len(longitudes)))**2 for x in longitudes) / len(longitudes))**0.5 if longitudes else 0,
                'min': min(longitudes) if longitudes else 0,
                'max': max(longitudes) if longitudes else 0
            }
        
        print("\nLongitud de los mensajes preprocesados:")
        for etiqueta, stats in longitud_prep_stats.items():
            print(f"{etiqueta}: {stats}")
        
        self.visualizador.visualizar_longitud_preprocesado(self.mensajes)
    
    def analizar_palabras_frecuentes_preprocesado(self, n=20) -> None:
        if not self.mensajes:
            print("No hay mensajes cargados")
            return
        
        def get_words(text):
            if not text or not isinstance(text, str):
                return []
            return text.split()
        
        ham_words = []
        spam_words = []
        
        for mensaje in self.mensajes:
            if mensaje.etiqueta == 'ham':
                ham_words.extend(get_words(mensaje.texto_preprocesado))
            elif mensaje.etiqueta == 'spam':
                spam_words.extend(get_words(mensaje.texto_preprocesado))
        
        ham_word_counts = Counter(ham_words)
        spam_word_counts = Counter(spam_words)
        
        self.resultado.palabras_frecuentes_ham_prep = ham_word_counts.most_common(n)
        self.resultado.palabras_frecuentes_spam_prep = spam_word_counts.most_common(n)
        
        print("\nTop 20 palabras después del preprocesamiento:")
        
        self.visualizador.visualizar_palabras_frecuentes(self.resultado, preprocesado=True)
    
    def generar_wordcloud_preprocesado(self) -> None:
        if not self.mensajes:
            print("No hay mensajes cargados")
            return
        
        print("\nWordCloud con texto preprocesado:")
        self.visualizador.visualizar_wordcloud(self.mensajes, preprocesado=True)
    
    def dividir_datos(self):
        if not self.mensajes:
            print("No hay mensajes cargados para dividir")
            return
        
        random.seed(self.random_seed)
        
        mensajes_spam = [m for m in self.mensajes if m.etiqueta == 'spam']
        mensajes_ham = [m for m in self.mensajes if m.etiqueta == 'ham']
        
        random.shuffle(mensajes_spam)
        random.shuffle(mensajes_ham)
        
        n_spam_test = int(len(mensajes_spam) * self.test_size)
        n_ham_test = int(len(mensajes_ham) * self.test_size)
        
        spam_test = mensajes_spam[:n_spam_test]
        spam_train = mensajes_spam[n_spam_test:]
        
        ham_test = mensajes_ham[:n_ham_test]
        ham_train = mensajes_ham[n_ham_test:]
        
        self.mensajes_train = spam_train + ham_train
        self.mensajes_test = spam_test + ham_test
        
        random.shuffle(self.mensajes_train)
        random.shuffle(self.mensajes_test)
        
        print(f"\nDivisión de datos completada:")
        print(f"  - Conjunto de entrenamiento: {len(self.mensajes_train)} mensajes")
        print(f"    - SPAM: {len([m for m in self.mensajes_train if m.etiqueta == 'spam'])}")
        print(f"    - HAM: {len([m for m in self.mensajes_train if m.etiqueta == 'ham'])}")
        print(f"  - Conjunto de prueba: {len(self.mensajes_test)} mensajes")
        print(f"    - SPAM: {len([m for m in self.mensajes_test if m.etiqueta == 'spam'])}")
        print(f"    - HAM: {len([m for m in self.mensajes_test if m.etiqueta == 'ham'])}")
    
    def entrenar_modelo(self):
        if not self.mensajes_train:
            print("Dividiendo los datos primero...")
            self.dividir_datos()
        
        print("\nEntrenando modelo bayesiano...")
        self.clasificador.entrenar(self.mensajes_train)
        print("Entrenamiento completado.")
    
    def evaluar_modelo(self):
        if not hasattr(self.clasificador, 'vocabulario') or not self.clasificador.vocabulario.prob_palabra_spam:
            print("El modelo debe ser entrenado primero")
            return
        
        if not self.mensajes_test:
            print("No hay datos de prueba disponibles")
            return
        
        print("\nEvaluando modelo con conjunto de prueba...")
        self.clasificador.metricas = self.clasificador.evaluar(self.mensajes_test)
        
        print("\nResultados de la evaluación:")
        print(f"Matriz de confusión:")
        matriz = self.clasificador.metricas.matriz_confusion
        print(f"               | Pred HAM  | Pred SPAM")
        print(f"---------------|-----------|-----------")
        print(f"Real HAM       | {matriz['ham']['ham']} | {matriz['ham']['spam']}")
        print(f"Real SPAM      | {matriz['spam']['ham']} | {matriz['spam']['spam']}")
        
        print(f"\nPrecisión: {self.clasificador.metricas.precision:.4f}")
        print(f"Recall: {self.clasificador.metricas.recall:.4f}")
        print(f"F1-Score: {self.clasificador.metricas.f1_score:.4f}")
        
        self.visualizador.visualizar_matriz_confusion(self.clasificador.metricas)
    
    def optimizar_threshold(self):
        if not hasattr(self.clasificador, 'vocabulario') or not self.clasificador.vocabulario.prob_palabra_spam:
            print("El modelo debe ser entrenado primero")
            return
        
        if not self.mensajes_test:
            print("No hay datos de prueba disponibles")
            return
        
        print("\nOptimizando threshold...")
        self.clasificador.optimizar_threshold(self.mensajes_test)
        
        self.visualizador.visualizar_thresholds(self.clasificador.metricas)
        
        self.visualizador.visualizar_palabras_predictivas(self.clasificador.vocabulario)
    
    def ejecutar_analisis_completo(self, ruta_archivo: str) -> None:
        print("="*80)
        print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
        print("="*80)
        
        self.cargar_datos(ruta_archivo)
        self.mostrar_mensajes_aleatorios()
        print(f"\nCantidad total de mensajes: {len(self.mensajes)}")
        self.analizar_distribucion()
        self.analizar_longitud_mensajes()
        self.analizar_palabras_frecuentes()
        self.generar_wordcloud()
        
        print("\n"+"="*80)
        print("LIMPIEZA DE DATOS")
        print("="*80)
        
        self.aplicar_preprocesamiento()
        print(f"\nCantidad total de mensajes después del preprocesamiento: {len(self.mensajes)}")
        self.analizar_longitud_preprocesado()
        self.analizar_palabras_frecuentes_preprocesado()
        self.generar_wordcloud_preprocesado()
        
        print("\n"+"="*80)
        print("MODELO BAYESIANO")
        print("="*80)
        
        self.dividir_datos()
        self.entrenar_modelo()
        
        print("\n"+"="*80)
        print("PRUEBAS DE RENDIMIENTO")
        print("="*80)
        
        self.evaluar_modelo()
        self.optimizar_threshold()
        
        print("\n"+"="*80)
        print("DISCUSIÓN DE RESULTADOS")
        print("="*80)
        
        self.discutir_resultados()
    
    def discutir_resultados(self):
        if not hasattr(self.clasificador, 'metricas') or not self.clasificador.metricas.matriz_confusion:
            print("No hay resultados para discutir. Ejecute el análisis completo primero.")
            return
        
        print("\nDiscusión de resultados:")
        
        precision = self.clasificador.metricas.precision
        recall = self.clasificador.metricas.recall
        f1 = self.clasificador.metricas.f1_score
        
        print(f"\n1. Interpretación de métricas:")
        print(f"   - Precisión ({precision:.4f}): Indica que de todos los mensajes clasificados como SPAM,")
        print(f"     aproximadamente el {precision*100:.1f}% son realmente SPAM.")
        print(f"   - Recall ({recall:.4f}): Muestra que nuestro modelo detecta correctamente")
        print(f"     el {recall*100:.1f}% de todos los mensajes SPAM reales.")
        print(f"   - F1-Score ({f1:.4f}): Es la media armónica entre precisión y recall,")
        print(f"     proporcionando una métrica única para evaluar el equilibrio entre ambas.")
        
        matriz = self.clasificador.metricas.matriz_confusion
        falsos_positivos = matriz['ham']['spam']
        falsos_negativos = matriz['spam']['ham']
        
        print(f"\n2. Análisis de errores:")
        print(f"   - Falsos positivos (HAM clasificado como SPAM): {falsos_positivos}")
        print(f"   - Falsos negativos (SPAM clasificado como HAM): {falsos_negativos}")
        
        print(f"\n3. Impacto de las decisiones tomadas:")
        
        print(f"   a) Preprocesamiento:")
        print(f"      - Eliminación de stopwords: Redujo el ruido en los datos y permitió")
        print(f"        centrarse en palabras más informativas para la clasificación.")
        print(f"      - Stemming: Unificó las diferentes formas de las palabras, reduciendo")
        print(f"        la dimensionalidad del vocabulario y mejorando la generalización.")
        print(f"      - El preprocesamiento redujo significativamente la longitud promedio")
        print(f"        de los mensajes, eliminando información redundante o no relevante.")
        
        print(f"   b) Modelado:")
        print(f"      - Suavizado de Laplace: Evitó probabilidades cero para palabras no vistas,")
        print(f"        mejorando la robustez del modelo ante vocabulario desconocido.")
        print(f"      - La selección del threshold óptimo ({self.clasificador.metricas.mejor_threshold})")
        print(f"        permitió equilibrar la precisión y el recall según las necesidades.")
        
        print(f"\n4. Interpretación global:")
        if f1 > 0.9:
            rendimiento = "excelente"
        elif f1 > 0.8:
            rendimiento = "muy bueno"
        elif f1 > 0.7:
            rendimiento = "bueno"
        elif f1 > 0.6:
            rendimiento = "aceptable"
        else:
            rendimiento = "mejorable"
        
        print(f"   - El modelo demuestra un rendimiento {rendimiento} en la clasificación de SPAM/HAM.")
        
        palabras_pred = self.clasificador.vocabulario.palabras_mas_predictivas[:5]
        print(f"\n5. Características más predictivas:")
        print(f"   - Las palabras con mayor poder predictivo para SPAM son:")
        for palabra, prob in palabras_pred:
            print(f"     * '{palabra}' (P(SPAM|palabra) = {prob:.4f})")
        
        print(f"\n6. Conclusiones finales:")
        print(f"   - El modelo bayesiano ofrece un buen equilibrio entre interpretabilidad y rendimiento.")
        print(f"   - La aproximación probabilística permite identificar claramente qué características")
        print(f"     son más informativas para la clasificación de mensajes.")
        print(f"   - El enfoque implementado es computacionalmente eficiente y escalable,")
        print(f"     adecuado para sistemas de filtrado de SPAM en tiempo real.")
        
        print(f"\n7. Posibles mejoras futuras:")
        print(f"   - Incorporar análisis de n-gramas para capturar relaciones entre palabras.")
        print(f"   - Implementar técnicas más avanzadas de vectorización como TF-IDF.")
        print(f"   - Evaluar el uso de clasificadores más complejos como SVM o redes neuronales.")
        print(f"   - Incluir análisis de características estructurales (p. ej., presencia de URLs,")
        print(f"     formato del mensaje, etc.) para complementar el análisis textual.")
    
    def predecir_texto(self, texto):
        if not hasattr(self.clasificador, 'vocabulario') or not self.clasificador.vocabulario.prob_palabra_spam:
            print("El modelo debe ser entrenado primero")
            return None, 0.0, []
        
        texto_preprocesado = self.preprocesar_texto(texto)
        
        etiqueta, prob_spam, palabras_predictivas = self.clasificador.predecir_mensaje(
            texto_preprocesado, 
            self.clasificador.metricas.mejor_threshold if hasattr(self.clasificador.metricas, 'mejor_threshold') else 0.5
        )
        
        return etiqueta, prob_spam, palabras_predictivas


class PrediccionInteractiva:
    def __init__(self, analizador: AnalizadorSpamHam):
        self.analizador = analizador
    
    def ejecutar(self):
        if not hasattr(self.analizador.clasificador, 'vocabulario') or not self.analizador.clasificador.vocabulario.prob_palabra_spam:
            print("Error: El modelo no ha sido entrenado. Ejecute el análisis completo primero.")
            return
        
        print("\n"+"="*80)
        print("MÓDULO DE PREDICCIÓN INTERACTIVA")
        print("="*80)
        print("Ingrese un texto para verificar si es SPAM o HAM (o 'salir' para terminar)")
        
        while True:
            texto = input("\nTexto a analizar: ").strip()
            
            if texto.lower() == 'salir':
                print("¡Hasta luego!")
                break
            
            if not texto:
                print("Por favor ingrese un texto válido")
                continue
            
            etiqueta, prob_spam, palabras_predictivas = self.analizador.predecir_texto(texto)
            
            print("\nResultado del análisis:")
            print(f"  - Clasificación: {etiqueta.upper()}")
            print(f"  - Probabilidad de SPAM: {prob_spam:.4f} ({prob_spam*100:.2f}%)")
            
            if palabras_predictivas:
                print("\n  - Palabras con mayor poder predictivo:")
                for palabra, prob in palabras_predictivas:
                    print(f"    * '{palabra}': {prob:.4f}")
            else:
                print("\n  - No se encontraron palabras con alto poder predictivo en este texto.")


if __name__ == "__main__":
    analizador = AnalizadorSpamHam()
    
    analizador.ejecutar_analisis_completo('spam_ham.csv')
    
    predictor = PrediccionInteractiva(analizador)
    predictor.ejecutar()