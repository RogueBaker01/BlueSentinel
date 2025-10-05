# BlueSentinel 🦈

## Summary

To solve the mystery of where and why sharks hunt, we developed **Blue Sentinel**, a solution combining a smart tag and a powered predictive model. The tag uses an accelerometer to detect the unique signature of a hunt, triggering a camera to capture images of the prey, while also recording temperature and salinity. This system addresses the challenge by sending this invaluable "ground-truth" data to a cloud platform, where it's integrated with NASA's oceanographic data (SST, chlorophyll levels). Our model learns these correlations to predict future feeding hotspots. This is vital because for the first time, we can move from just tracking sharks to understanding their role in the food web, creating a powerful tool to protect marine biodiversity and inform conservation policy.

---

## Project Details

### What it does and how it works?

Blue Sentinel is a comprehensive, three-part ecosystem designed to move from simply tracking sharks to deeply understanding and predicting their feeding behavior.

#### 1. The "Sentinel Tag" (Data Collection)

This smart, non-invasive tag is attached to a shark. Its core is an Inertial Measurement Unit (IMU) that continuously monitors the animal's movement. Our onboard algorithm is trained to recognize a specific **"hunting signature"**—a pattern of high-speed acceleration and rapid gyroscopic change. This trigger activates a micro-camera to capture a 5-photo burst, providing visual evidence of the prey. Simultaneously, the tag logs crucial environmental data: GPS location (when surfaced), water temperature, and conductivity (salinity). Upon surfacing, the tag transmits this compact data package—images, location, and sensor readings—via a satellite modem.

#### 2. The Predictive Brain

The data is received by our cloud-based backend. Here, a Computer Vision Model analyzes the photos to identify the prey species. This confirmed hunting event—a specific prey at a specific time, location, and in specific environmental conditions—becomes a high-quality "ground-truth" data point. This point is then fed into our primary machine learning model, which correlates it with large-scale NASA satellite data (Sea Surface Temperature, Chlorophyll-a from MODIS/VIIRS, and altimetry). Over time, the model learns the complex relationships between oceanic conditions visible from space and actual predatory events, allowing it to generate a predictive heatmap of likely feeding hotspots.

#### 3. The Mobile App (User Interface & Engagement)

The final piece is our mobile application, which serves as the window into our project for both scientists and the public. It connects to our cloud platform and visualizes the data in an intuitive way. Users can track individual sharks on a map, see pins indicating confirmed hunting events (and view the associated images), and observe the environmental data from the tag. Most importantly, the app overlays our predictive heatmap, showing where feeding activity is most likely to occur. It also includes an educational section with facts about species, their importance, and conservation efforts.

---

## Benefits and Intended Impact

The benefits and impact of BlueSentinel are multi-layered:

* **Scientific Breakthrough**: It provides an unprecedented, empirical dataset on predator-prey interactions in the wild, answering fundamental questions about marine food webs.
* **Enhanced Conservation**: By identifying critical feeding habitats, the project can directly inform the creation and management of Marine Protected Areas (MPAs). Understanding a species' diet is crucial for its protection.
* **Predictive Power**: For the first time, we can move from reactive monitoring to proactive prediction, anticipating where shark-human encounters might be more likely or where fishing activities should be limited to protect the ecosystem.
* **Public Engagement**: The mobile app transforms complex scientific data into an engaging story. It aims to shift the public perception of sharks from fear to fascination, fostering a global community of ocean advocates.

---

## Tools, Coding Languages, Hardware, and Software

### Hardware (The Tag Prototype)

* **Microcontroller**: ESP32 for its processing power, low energy consumption, and integrated Wi-Fi/Bluetooth for testing.
* **Sensors**: MPU-6050 (IMU), waterproof DS18B20 (Temperature), a custom conductivity sensor, and a U-blox GPS module.
* **Camera**: ArduCam Mini camera module.
* **Communication**: Iridium or Argos satellite modem module for data transmission.

### Software (Backend & AI)

* **Language**: Python.
* **Libraries**: TensorFlow/PyTorch for the Computer Vision and ML models; Pandas and Scikit-learn for data analysis; Flask/Django to build the API for the mobile app.
* **Database**: PostgreSQL with PostGIS for handling geospatial data.
* **NASA Data**: Accessed via NASA's Earthdata Search and APIs.

### Software (Mobile App)

* **Framework**: React Native (iOS/Android) development.
* **Mapping**: Mapbox API for its powerful data visualization and customization capabilities.

---

## Use of Artificial Intelligence (AI)

Our project, Blue Sentinel, leverages Artificial Intelligence in two distinct and fundamental ways:

1.  As a tool to accelerate our development and documentation process during the hackathon.
2.  As a core, functional component of our proposed solution.

We believe in transparently acknowledging the role of AI in our workflow. We used **Gemini** and **ChatGPT** to assist in the development of this project.

# Datasets: 
https://drive.google.com/drive/folders/1UCJoLE58jK5aSTLOnBFv3-mU-YNOBsuV?usp=drive_link
# FIGMA: 
https://www.figma.com/design/zvx7sj2ayWwoHyfvqjbmGb/BlueSentinel?node-id=0-1&p=f&t=LKrjPgpk69bkLdVf-0
