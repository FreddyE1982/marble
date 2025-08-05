# Neural Pathway GUI Design

## Overview
The Neural Pathway Inspector enables interactive exploration of signal routes
inside a MARBLE core directly from the Streamlit playground. Users specify a
start and end neuron and the inspector computes the pathway between them using
a tensor based breadth first search.

## CPU/GPU Support
Edges are converted to an adjacency matrix stored in a Torch tensor. The search
runs on ``cuda`` when available and falls back to ``cpu`` otherwise. The code is
identical across devices so results remain consistent independent of hardware.

## GUI Workflow
1. The NB Explorer tab hosts a *Neural Pathway Inspector* expander.
2. Users enter start and end neuron identifiers and press **Find Pathway**.
3. The inspector displays the list of neuron IDs forming the path and renders a
   Plotly graph with the pathway highlighted.
4. Missing connections produce a clear warning.

## Visualization
The plot highlights nodes and edges belonging to the path in red while other
connections remain blue/gray. Layout options include spring and circular
arrangements inherited from ``networkx``.

## Testing
Unit tests verify the search on CPU and GPU and ensure the figure generator
returns a ``plotly.graph_objs.Figure``. Integration tests load the Streamlit
playground, invoke the inspector and confirm a Plotly chart is emitted.
