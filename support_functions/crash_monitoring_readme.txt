Note: the following instructions must be followed on the computer running lifetime testing to send restart notifications (in case of power bump) to Brianna.

*STEP 1*: Save NotifyStartup.ps1 to the folder C:\\Scripts

When run, this code will send a notification to the following webhook, which sends a slack message to Brianna with notification that the computer has restarted

[the program uses this slack webhook: https://hooks.slack.com/services/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]

*STEP 2*: Setup a task to run the program on reboot

1. Open task scheduler, and click "Create Task..."
2. Under "General":
	- Name the task: Slack Notification on Boot
	- Select "Run whether user is logged on or not"
	- Check "Run with highest privileges"
	- Configure for your version of Windows
3. Under "Triggers":
	- Click "New..."
	- Begin the task "At startup"
4. Under "Actions":
	- Click "New..."
	- Use the action "Start a program"
	- Under "Program/script", enter: powershell.exe
	- Add arguments: -ExecutionPolicy Bypass -File "C:\Scripts\NotifyStartup.ps1"
5. Under "Settings":
	- Check "Allow task to be run on demand"
	- Check "If the task fails, restart every:"
		- enter 1 minute, and 3 restart attempts
	- Check "Stop the task if it runs longer than:"
		- enter 1 hour
6. Click "Ok"