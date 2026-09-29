# Flapping Noise Remover 

Este script de Python elimina el ruido mecánico asimétrico de aleteo y las altas resonancias del motor en grabaciones de alas batientes a partir de las rpm's en tiempo real y cancelación de ruido activa. 


## Características Técnicas

* Cruza el audio del micrófono principal, un micrófono de referencia a 50 cm de distancia y la telemetría del motor extraída de un archivo CSV (MAVLink).
* Incluye estimación de ruido por ciclo de aleteo.
  
## Instalación y Requisitos

Asegúrate de tener instalado Python 3 y las siguientes librerías:

```bash
pip install numpy scipy pandas
