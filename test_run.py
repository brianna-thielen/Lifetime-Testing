import requests
import traceback
import datetime

SLACK_WEBHOOK_URL = 'https://hooks.slack.com/services/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'

def notify_slack(message):
    payload = {'text': message}
    requests.post(SLACK_WEBHOOK_URL, json=payload)

def write_heartbeat():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("heartbeat.txt", "w") as f:
        f.write(f"Last heartbeat: {now}\n")

def main_loop():
    # Your main script logic here
    a = 0
    while True:
        # Simulated code
        print (f'loop {a}')

        a = a + 1

        write_heartbeat()

        if a > 10:
            raise ValueError("Simulated error for testing Slack notification")
        pass

if __name__ == '__main__':
    try:
        main_loop()
    except Exception as e:
        error_msg = f"Lifetime testing script crashed at {datetime.datetime.now()}:\n{traceback.format_exc()}"
        print(error_msg)
        notify_slack(error_msg)
