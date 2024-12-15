# Prototipo de funciones auxiliares para el proyecto de IA

'''
def save_checkpoint(model, optimizer, epoch, loss, ..., checkpoint_dir='./checkpoints'):
    """Guarda un checkpoint del modelo, los parametros de la función cambiarian dependiendo del modelo"""
    
    # Crear directorio si no existe
    # Guardar el modelo, el optimizador, el epoch, la loss, etc.
    # Guardar el checkpoint en un archivo .pth
    # Retornar la ruta al archivo
    # 
 '''

'''
def setup_model():
    """Resliza la configuración nesecaria para cargar un modelo y realizar el entrenamiento"""
    # Cargar dataset
    # Cargar modelo
    # Limpiar memoria
    # crear backup de checkpoints y modelos entrenados previamente si es necesario
    

def train_model("parametros"):
    """Entrena el modelo"""
    # Verificar si existe un checkpoint previo
    # Cargar modelo
    # Entrenar modelo
    # Guardar checkpoints
    # Guardar modelo entrenado
    # Limpiar memoria
    # Retornar modelo entrenado

def test_model("parametros"):
    """Realiza la evaluación del modelo"""
    # Cargar modelo
    # Cargar checkpoints
    # Evaluar modelo
    # Limpiar memoria
    # Retornar resultados

def predict("parametros"):
    """Realiza predicciones con el modelo"""
    # Cargar modelo
    # Cargar checkpoints
    # Realizar predicciones
    # Limpiar memoria
    # Retornar resultados


def main():
    """Función principal"""
    # Configurar modelo
    # entrenar el modelo
    # realizar pruebas
'''