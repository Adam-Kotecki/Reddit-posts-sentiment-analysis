import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import base64

# Sample data for demonstration
# Replace this with your actual data
data = {
    'Category': ['A', 'B', 'C', 'D', 'E', 'F'],
    'Value': [10, 20, 15, 25, 30, 35]
}
df = pd.DataFrame(data)

# Function to generate chart and save as PNG
def generate_chart_and_save_png(category):
    fig = px.bar(df, x='Category', y='Value')
    fig.update_layout(title=f'Chart for {category}')
    fig.write_image(f'chart_{category}.png')

# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div(style={'background': 'linear-gradient(to bottom, #7028a9, #7e45a8)', 'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}, children=[
    html.H1('Sentiment Analysis', style={'color': 'white', 'textAlign': 'center'}),
    *[html.Div(html.Img(id=f'chart{i}', src=''), style={'margin': '10px'}) for i in range(1, 7)]
])

# Callback to update charts
@app.callback(
    [Output(f'chart{i}', 'src') for i in range(1, 7)],
    [Input(f'chart{i}', 'id') for i in range(1, 7)]
)
def update_charts(*args):
    for i in range(1, 7):
        generate_chart_and_save_png(i)
    return [f'data:image/png;base64,{base64.b64encode(open(f"chart_{i}.png", "rb").read()).decode()}' for i in range(1, 7)]

if __name__ == '__main__':
    app.run_server(debug=True)

