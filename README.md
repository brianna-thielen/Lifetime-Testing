# Lifetime-Testing
CV, EIS, and VT testing plus data analysis for accelerated lifetime testing

Instructions for adding new samples:


Instructions for removing samples from test:


Instructions for setting up slack webhook:
1. Go to https://api.slack.com/apps
2. Click “Create New App” → From scratch.
3. Give it a name and select your workspace.
4. Go to “Incoming Webhooks” in the left menu.
5. Toggle Activate Incoming Webhooks to ON.
6. Click “Add New Webhook to Workspace” and select the desired channel.
7. Copy the Webhook URL — it’ll look like:
    https://hooks.slack.com/services/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
8. Paste this into equipment.json under "Slack" / "webhook", replacing the existing webhook.


Instructions for setting up arduino:
- To replicate this setup, this will require some reverse engineering of the hardware that's currently used on the existing setup, as this isn't documented
- The code that's used to program the arduino is saved under support_functions as i2c_RH_Temp_muxed2_fast.ino