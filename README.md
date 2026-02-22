# Lifetime-Testing
Electrode testing (EIS and VT), IDE testing (EIS), and humidity sensor monitoring (I2C and resistive) plus data analysis for accelerated lifetime testing

Instructions for starting testing:
1. Ensure all hardware is connected and turned on
2. Open Visual Studio Code to Lifetime-Testing repository
3. Open Intan RHX software
4. Select the RHS 128ch Stim/Recording Controller
5. Select the 30 kHz sample rate and +/- 2.550 mA stimulation range
6. Click Network -> Remote TCP Control
7. In the "Commands" tab, click "Connect"
8. In the "Data Output" tab, under "Waveform Output", click "Connect"
9. In Visual Studio Code, run the program run_lifetime_testing.py

Instructions for adding new samples:
1. Pause testing by pressing "ctrl-c"
    Note: if "CONTINUOUS_TESTING" is set to "True" in "run_lifetime_testing.py", you will need to re-connect the Intan and run "pause_lifetime_testing.py"
    For IDEs:
        2. Connect each sample via a Seriemas3 custom PCB (tapeout package under equipment/Seriemas3_RevX01_Tapeout_12142024_TIME.zip) connected to the switcher
        3. Create a new .json file for the sample group, following "LCP Pt Grids.json" as a template, and save it in test_information/samples
    For Stimulating Electrodes:
        2. Connect each sample to an Intan channel, with return electrodes connected to Intan ground
        3. Create a new .json file for the sample group, following "LCP IDEs 25um.json" as a template, and save it in test_information/samples
    For Recording Electrodes:
        2. Connect either as IDEs (to measure with the LCR meter) or as stimulating electrodes with zero stimulation amplitude (to measure with the Intan)
        3. Create a new .json file for the sample group (according to the appropriate connection protocol), and save it in test_information/samples
    Humidity Sensors (capacitive, I2C):
        2. Replicate the I2C scanner hardware on the existing Arduino UNO, connect to a new Arduino Uno, connect the sensors to the Arduino, connect to the computer
            - The hardware is not documented, so this will require reverse engineering
            - The code that's used to program the arduino is saved under support_functions as i2c_RH_Temp_muxed2_fast.ino
            - You need to determine the port for the new Arduino and enter it in the json file
        3. Create a new .json file for the sample group, following "LCP Encapsulation Capacitive.json" as a template, and save it in test_information/samples
    Humidity Sensors (resistive, impedance):
        2. Connect each sample via a Seriemas3 custom PCB (tapeout package under equipment/Seriemas3_RevX01_Tapeout_12142024_TIME.zip) connected to the switcher
        3. Create a new .json file for the sample group, following "LCP Encapsulation Resistive.json" as a template, and save it in test_information/samples
4. Re-start testing by running the program run_lifetime_testing.py

Instructions for removing samples from test:
1. Pause testing by pressing "ctrl-c"
    Note: if "CONTINUOUS_TESTING" is set to "True" in "run_lifetime_testing.py", you will need to re-connect the Intan and run "pause_lifetime_testing.py"
2. Disconnect samples
3. Open the sample group information under test_information/samples/[group].json
4. To remove a single device from a group, add the name of the sample to the "broken devices" list
   To remove the whole group, list the current date and time in the format "YYYY-MM-DD HH:MM" (24 hour format) under "end_date"
5. Re-connect the Intan, and run "run_lifetime_testing.py"

Instructions for setting up slack webhook:
1. Go to https://api.slack.com/apps
2. Click “Create New App” → From scratch.
3. Give it a name and select your workspace.
4. Go to “Incoming Webhooks” in the left menu.
5. Toggle Activate Incoming Webhooks to ON.
6. Click “Add New Webhook to Workspace” and select the desired channel.
7. Copy the Webhook URL, and save it into a new file under test_information/slack.json, with the key "webhook" (example below)
    slack.json:
    {
        "webhook": "[webhook url]"
    }

Instructions for setting up Github upload:
1. Sync the Github repository to your computer
2. Save the directory under test_information/github.json, with the key "path" (example below)
    github.json:
    {
        "path": "C:/Repo-Directory"
    }
3. If push fails, ensure you're logged in by testing a push on the command line

Note: slack.json and github.json are automatically added to .gitignore for privacy