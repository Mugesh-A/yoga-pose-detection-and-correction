# Yoga Pose Detection and Correction

<h1>Description</h1>

Welcome to my new deep learning project, **"Yoga Pose Detection and Correction."** As the name indicates, this project can identify the yoga pose you are performing in front of your webcam using an image sensor.<br>

The project consists of three Python scripts:<br>
- **Data Collection**<br>
- **Data Training**<br>
- **Inference Script**<br>

Each script serves its respective purpose.<br>

In this project, I utilized **MediaPipe** for pose detection, enabling the identification of human body poses. Following that, I built a model using a simple Dense network with **Keras** and trained it on the collected data. Finally, I ran the inference script to make predictions.<br>

<h1>Requirements</h1>

<code>pip install mediapipe</code><br>
<code>pip install keras</code><br>
<code>pip install tensorflow</code><br>
<code>pip install opencv-python</code><br>
<code>pip install numpy</code><br>

<h1>How to Run?</h1>

<h2>Adding Data</h2>

To add data, you need to run <b>python data_collection.py</b> and provide the name of the asana you wish to add.<br>

<h2>Training</h2>

To train the model on the newly added data, simply run <b>python data_training.py</b>.<br>

<h2>Running</h2>

To execute the model and see predictions, run <b>python inference.py</b>. A new window will appear, displaying the ongoing predictions.<br>
  
<h1>Video</h1>

Video link: [Watch here](https://www.youtube.com/watch?v=6w3g_33wMOs)<br>

---

Let me know if there's anything else you need!
