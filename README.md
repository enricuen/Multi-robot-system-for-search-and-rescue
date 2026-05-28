# Multi-robot-system-for-search-and-rescue
Este repositorio contiene un nodo de ROS 2 en Python que simula e implementa un enjambre de robots colaborativos diseñado para misiones de búsqueda, localización y asistencia a víctimas.
El sistema combina navegación autónoma descentralizada (**Nav2**), visión artificial mediante Deep Learning (**YOLOv8**), detección reactiva de obstáculos con LIDAR y una máquina de estados para la coordinación táctica de múltiples agentes (exploradores y ambulancia).

## Características principales

* **Arquitectura multiagente:** Despliegue de un enjambre escalable coordinado mediante una única máquina de estados descentralizada.
* **Visión artificial local:** Integración de **YOLOv8** en cada nodo para la detección visual de víctimas (clase `person`) en tiempo real, generando un error de seguimiento (*Homing*).
* **Navegación autónoma y evasión:** Uso de **Nav2** para la planificación de rutas globales hacia objetivos y un sistema de evasión puramente reactivo (LIDAR) como freno de emergencia anti-colisiones.
* **Protocolos interactivos de rescate:**
  
    * *Aseguramiento:* Formación de un perímetro de vigilancia alrededor de víctimas a salvo.

      
    <img width="347" height="250" alt="Screenshot from 2026-05-28 20-55-04" src="https://github.com/user-attachments/assets/7bc745bc-1dc4-43b0-87d7-209bc871cf23" />

    * *Emergencia:* Despliegue automático y en solitario del "Robot ambulancia" hacia coordenadas de personas heridas.
    * 
      
    <img width="347" height="250" alt="Screenshot from 2026-05-28 20-08-24" src="https://github.com/user-attachments/assets/13000639-faff-4572-86e4-bf650f25eb1e" />

## Requisitos previos

Para ejecutar este proyecto, es necesario un entorno de ROS 2 configurado junto con bibliotecas de visión e IA. Se asume el uso de **ROS 2 Jazzy**.

### Bibliotecas del sistema y ROS 2
Se ha de tener instalados los siguientes paquetes de ROS 2:

```bash
sudo apt update
sudo apt install ros-jazzy-rclpy \
                 ros-jazzy-geometry-msgs \
                 ros-jazzy-sensor-msgs \
                 ros-jazzy-nav2-msgs \
                 ros-jazzy-tf2-ros \
                 ros-jazzy-cv-bridge \
                 ros-jazzy-navigation2 \
                 ros-jazzy-nav2-bringup
```
### Bibliotecas de Python
El nodo requiere librerías para procesar las imágenes y ejecutar la red neuronal YOLO:

```bash
pip3 install opencv-python numpy ultralytics --break-system-packages
```

## Lanzar y ejecutar el comportamiento
* Paso 1: Lanzar la simulación

```bash
colcon build
source install/setup.bash
ros2 launch robot_bringup multi_robot_sim_nav2.launch.py
```
* Paso 2: Ejecutar el nodo del comportamiento
```bash
source install/setup.bash
ros2 run robot_bringup esquinas.py
```
* Interacción con el sistema

Una vez en marcha, los robots se desplegarán de forma escalonada hacia sus esquinas de búsqueda.

Al llegar frente a la víctima, el sistema pausará la terminal y preguntará:

**¿La persona está 'herida' o 'a salvo'?:**

A esto se debe contestar ```herida``` o ```a salvo```

