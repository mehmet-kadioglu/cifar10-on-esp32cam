# ESP32CAM Cifar10 Project But Used Two Images, Dog and Cat

- After training model on edge impulse run this code for including library to platformio in lib folder: `~/.platformio/penv/bin/pio lib install lib/cifar10_final-v3.zip`
- If using windows run like this: `C:\path\to\pio\exe\pio.exe lib install lib\deployedlib.zip`
- After importing library to find examples codes, go into .pio/libdeps/esp32cam/Cifar10_final_inferencing/examples/esp32/esp32_camera then find .ino file and paste it into main.cpp
- After pasting code select microcontroller model accordingly
- After building code if code is giving error then move this function definiton to top of the file under the function definitions: `static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr);`
- If driver is not loaded accordingly download and install CH340 driver
- After uploading code if platformio serial monitor is not working correctly then use ArduinoIDE serial monitor

## Used embedded device parts:
<img src="img/img1.jpeg" style="width:100%"/>
<img src="img/img2.jpeg" style="width:100%"/>
<img src="img/img3.jpeg" style="width:100%"/>
<img src="img/img4.jpeg" style="width:100%"/>