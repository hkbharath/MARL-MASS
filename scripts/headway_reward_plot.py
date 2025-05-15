import numpy as np
import json

# Function definition
def compute_y(x):
    return -np.log(x / (0.5 * 25))  # Equivalent to -log(x / 12.5)

# Generate x values
x_log = np.logspace(-2, np.log10(0.1), num=5)
x_lin = np.arange(0.1001, 20.1, 0.1)
x_vals = np.unique(np.concatenate((x_log, x_lin)))

# Create data array
data = [{"x": float(x), "y": float(compute_y(x))} for x in x_vals]

# Define Vega spec
vega_spec = {
    "$schema": "https://vega.github.io/schema/vega/v5.json",
    "description": "Vega chart of y = -log(x / 12.5) with log-linear x scale and grid lines.",
    "width": 750,
    "height": 350,
    "padding": 5,
    "data": [
        {
            "name": "table",
            "values": data
        }
    ],
    "config": {
        "axis": {
            "titleFontSize": 32,
            "titleFontWeight": "normal",
            "labelFontSize": 24,
        }
    },
    "scales": [
        {
            "name": "xscale",
            "type": "linear",
            "domain": {"data": "table", "field": "x"},
            "range": "width"
        },
        {
            "name": "yscale",
            "type": "linear",
            "domain": {"data": "table", "field": "y"},
            "range": "height"
        }
    ],
    "axes": [
        {
            "orient": "bottom",
            "scale": "xscale",
            "title": "Headway distance (m)",
            "grid": True,  # Enable grid lines on x-axis
            "tickCount": 5
        },
        {
            "orient": "left",
            "scale": "yscale",
            "title": "Headway reward - rₕ",
            "grid": True,  # Enable grid lines on y-axis
            "tickCount": 5
        }
    ],
    "marks": [
        {
            "type": "line",
            "from": {"data": "table"},
            "encode": {
                "enter": {
                    "x": {"scale": "xscale", "field": "x"},
                    "y": {"scale": "yscale", "field": "y"},
                    "stroke": {"value": "steelblue"},
                    "strokeWidth": {"value": 2}
                }
            }
        }
    ]
}

# Save to file
with open("results/headway_reward_plot_vega.json", "w") as f:
    json.dump(vega_spec, f, indent=2)

print("Vega-Lite spec saved to headway_reward_vega.json")
