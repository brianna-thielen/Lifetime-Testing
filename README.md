# Lifetime-Testing
CV, EIS, and VT testing plus data analysis for accelerated lifetime testing

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