$log = "C:\Scripts\bootlog.txt"
"[$(Get-Date)] Script started" | Out-File -Append $log

Start-Sleep -Seconds 20  # wait for network stack to come up

$webhookUrl = "https://hooks.slack.com/services/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"
$hostname = $env:COMPUTERNAME
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

$payload = @{
    text = "*$hostname* restarted at $timestamp (UTC)."
} | ConvertTo-Json -Compress

# Try sending up to 3 times
for ($i = 1; $i -le 3; $i++) {
    try {
        Invoke-RestMethod -Uri $webhookUrl -Method Post -Body $payload -ContentType 'application/json'
        "[$(Get-Date)] Slack message sent (try $i)" | Out-File -Append $log
        break
    } catch {
        "[$(Get-Date)] Attempt $i failed: $_" | Out-File -Append $log
        Start-Sleep -Seconds 10
    }
}