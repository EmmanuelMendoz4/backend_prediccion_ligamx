import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Cargar variables de entorno (para pruebas locales con .env)
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("No se encontró la variable de entorno DATABASE_URL")

def main():
    print("Iniciando proceso de actualización de datos para la Liga MX...")
    
    # 1. Asegurar el uso del conector psycopg2 en la URL de conexión
    if "postgresql://" in DATABASE_URL and "+psycopg2" not in DATABASE_URL:
        db_url = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://")
    else:
        db_url = DATABASE_URL

    engine = create_engine(db_url)

    try:
        # 2. Conexión de prueba y lectura del dataset actual en Supabase
        print("Conectando a Supabase para verificar el estado de 'dataset_maestro'...")
        df_actual = pd.read_sql("SELECT * FROM dataset_maestro", con=engine)
        print(f"Registros actuales en la base de datos: {len(df_actual)}")

        # 3. Cargar o recibir los nuevos partidos de la jornada reciente
        # Puedes estructurar tus nuevos partidos en un archivo CSV local (ej: 'jornada_nueva.csv')
        # que contenga exactamente las mismas columnas que requiere tu modelo y endpoints.
        archivo_nuevos_partidos = "jornada_nueva.csv"
        
        if os.path.exists(archivo_nuevos_partidos):
            print(f"Archivo '{archivo_nuevos_partidos}' encontrado. Procesando nuevos partidos...")
            df_nuevos = pd.read_csv(archivo_nuevos_partidos)
            
            if df_nuevos.empty:
                print("El archivo de la jornada nueva está vacío. No hay datos que actualizar.")
                return

            # Opcional: Validar que las columnas coincidan o limpiar nulos si es necesario
            # 4. Insertar los nuevos registros directamente en la tabla de Supabase
            df_nuevos.to_sql(
                'dataset_maestro', 
                con=engine, 
                if_exists='append',  # 'append' agrega los nuevos partidos al final de la tabla sin borrar lo anterior
                index=False
            )
            
            print(f"¡Éxito! Se agregaron {len(df_nuevos)} partidos nuevos a la base de datos en Supabase.")
            
            # Verificar el total actualizado
            df_verificacion = pd.read_sql("SELECT COUNT(*) as total FROM dataset_maestro", con=engine)
            print(f"Total de registros ahora en Supabase: {df_verificacion['total'].values[0]}")
            
        else:
            print(f"No se encontró el archivo '{archivo_nuevos_partidos}'. Si tienes una API externa o script de scraping, conéctalo aquí.")

        print("¡Proceso de sincronización finalizado correctamente!")

    except Exception as e:
        print(f"Ocurrió un error crítico durante la actualización en Supabase: {e}")
        raise e

if __name__ == "__main__":
    main()