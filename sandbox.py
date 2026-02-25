import plotly.graph_objects as go
fig = go.Figure(go.Scatter(y=[1,2,3]))
fig.write_image(f"kaleido_smoke.png", width=2544, height=1289)
print("done")