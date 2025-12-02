
import io
from collections import defaultdict

import pandas as pd
import streamlit as st
import networkx as nx


st.set_page_config(page_title="FRA Rail Shortest Path Solver", layout="wide")
st.title("FRA Rail Shortest Path Solver (ZIP → FRAARCID chain)")


@st.cache_data(show_spinner=False)
def load_fra_segments(file) -> pd.DataFrame:
    df = pd.read_csv(file, dtype=str)
    # Clean column names to avoid space issues
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("__", "_")
    )
    # Ensure MILES is numeric
    if "MILES" in df.columns:
        df["MILES"] = pd.to_numeric(df["MILES"], errors="coerce")
    else:
        raise ValueError("FRA file must contain a 'MILES' column.")
    return df


@st.cache_data(show_spinner=False)
def build_graph_and_zip_index(fra_df: pd.DataFrame):
    """
    Build:
    - Undirected graph with edge weights = MILES
    - zip_to_nodes: ZIP -> set(FRANODE)
    - edge_lookup: (min(u,v), max(u,v)) -> list of FRAARCID(s)
    """
    required_cols = {"FRAARCID", "FRFRANODE", "TOFRANODE", "MILES"}
    missing = required_cols - set(fra_df.columns)
    if missing:
        raise ValueError(f"Missing required FRA columns: {missing}")

    G = nx.Graph()
    zip_to_nodes = defaultdict(set)
    edge_lookup = defaultdict(list)

    # Try to infer column names for ZIPs after cleaning
    possible_start_zip_cols = [c for c in fra_df.columns if "Start" in c and "ZIP" in c]
    possible_end_zip_cols = [c for c in fra_df.columns if "End" in c and "ZIP" in c]

    if not possible_start_zip_cols or not possible_end_zip_cols:
        raise ValueError(
            "Could not find start/end ZIP columns. "
            "Make sure they contain 'Start'/'End' and 'ZIP' in their names."
        )

    start_zip_col = possible_start_zip_cols[0]
    end_zip_col = possible_end_zip_cols[0]

    for row in fra_df.itertuples(index=False):
        u = getattr(row, "FRFRANODE")
        v = getattr(row, "TOFRANODE")
        miles = getattr(row, "MILES")

        if pd.isna(miles):
            continue

        miles = float(miles)

        fraarcid = getattr(row, "FRAARCID")
        G.add_edge(u, v, weight=miles)

        # ZIP index
        start_zip = getattr(row, start_zip_col)
        end_zip = getattr(row, end_zip_col)

        if pd.notna(start_zip):
            zip_to_nodes[str(start_zip)].add(u)
        if pd.notna(end_zip):
            zip_to_nodes[str(end_zip)].add(v)

        key = tuple(sorted((u, v)))
        edge_lookup[key].append(fraarcid)

    return G, zip_to_nodes, edge_lookup


def solve_one_pair(G, zip_to_nodes, edge_lookup, start_zip, end_zip):
    """
    Return dict with:
    - start_zip, end_zip
    - status ("ok" or "error")
    - message (if error)
    - rail_miles
    - num_segments
    - fraarcid_chain (comma-separated)
    - frnode_chain (comma-separated)
    """
    start_zip_str = str(start_zip)
    end_zip_str = str(end_zip)

    start_nodes = list(zip_to_nodes.get(start_zip_str, []))
    end_nodes = set(zip_to_nodes.get(end_zip_str, []))

    if not start_nodes:
        return {
            "start_zip": start_zip_str,
            "end_zip": end_zip_str,
            "status": "error",
            "message": f"No nodes found for start ZIP {start_zip_str}",
            "rail_miles": None,
            "num_segments": None,
            "fraarcid_chain": None,
            "frnode_chain": None,
        }

    if not end_nodes:
        return {
            "start_zip": start_zip_str,
            "end_zip": end_zip_str,
            "status": "error",
            "message": f"No nodes found for end ZIP {end_zip_str}",
            "rail_miles": None,
            "num_segments": None,
            "fraarcid_chain": None,
            "frnode_chain": None,
        }

    try:
        # Multi-source Dijkstra from all start_nodes
        distances, paths = nx.multi_source_dijkstra(G, start_nodes, weight="weight")
    except nx.NetworkXNoPath:
        return {
            "start_zip": start_zip_str,
            "end_zip": end_zip_str,
            "status": "error",
            "message": "NetworkXNoPath during multi-source search.",
            "rail_miles": None,
            "num_segments": None,
            "fraarcid_chain": None,
            "frnode_chain": None,
        }

    # Find best reachable end node
    candidate_targets = end_nodes.intersection(distances.keys())
    if not candidate_targets:
        return {
            "start_zip": start_zip_str,
            "end_zip": end_zip_str,
            "status": "error",
            "message": "No reachable end nodes for this ZIP pair.",
            "rail_miles": None,
            "num_segments": None,
            "fraarcid_chain": None,
            "frnode_chain": None,
        }

    best_t = min(candidate_targets, key=lambda t: distances[t])
    best_dist = distances[best_t]
    path_nodes = paths[best_t]  # list of FRANODEs

    # Convert node path → FRAARCID chain
    fraarcids = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        key = tuple(sorted((u, v)))
        arcids = edge_lookup.get(key)
        if not arcids:
            fraarcids.append("MISSING")
        else:
            fraarcids.append(arcids[0])

    return {
        "start_zip": start_zip_str,
        "end_zip": end_zip_str,
        "status": "ok",
        "message": "",
        "rail_miles": round(best_dist, 3),
        "num_segments": len(fraarcids),
        "fraarcid_chain": " > ".join(fraarcids),
        "frnode_chain": " > ".join(path_nodes),
    }


def solve_od_pairs(G, zip_to_nodes, edge_lookup, od_df, start_col, end_col):
    results = []
    for row in od_df.itertuples(index=False):
        start_zip = getattr(row, start_col)
        end_zip = getattr(row, end_col)
        res = solve_one_pair(G, zip_to_nodes, edge_lookup, start_zip, end_zip)
        # Preserve any additional columns (like OD id)
        base_row = row._asdict()
        base_row.update(res)
        results.append(base_row)
    return pd.DataFrame(results)


st.sidebar.header("Step 1: Upload data")

fra_file = st.sidebar.file_uploader("Upload FRA segments CSV", type=["csv"])
od_file = st.sidebar.file_uploader("Upload ZIP OD pairs CSV", type=["csv"])

if fra_file is not None and od_file is not None:
    try:
        fra_df = load_fra_segments(fra_file)
    except Exception as e:
        st.error(f"Error loading FRA file: {e}")
        st.stop()

    st.success(f"Loaded FRA file with {len(fra_df):,} segments.")

    try:
        G, zip_to_nodes, edge_lookup = build_graph_and_zip_index(fra_df)
    except Exception as e:
        st.error(f"Error building graph/index: {e}")
        st.stop()

    st.write(f"Graph has **{G.number_of_nodes():,} nodes** and **{G.number_of_edges():,} edges**.")

    # Load OD file
    od_df = pd.read_csv(od_file, dtype=str)
    st.success(f"Loaded OD file with {len(od_df):,} OD pairs.")

    # Let user pick start/end ZIP columns
    st.subheader("Select ZIP columns from OD file")
    cols = od_df.columns.tolist()
    start_col = st.selectbox("Start ZIP column", cols, index=0)
    end_col = st.selectbox("End ZIP column", cols, index=1 if len(cols) > 1 else 0)

    if st.button("Run shortest path solver"):
        with st.spinner("Computing shortest rail paths..."):
            results_df = solve_od_pairs(
                G, zip_to_nodes, edge_lookup, od_df, start_col, end_col
            )

        st.subheader("Results")
        st.dataframe(results_df, use_container_width=True)

        # Download
        csv_buf = io.StringIO()
        results_df.to_csv(csv_buf, index=False)
        st.download_button(
            label="Download results as CSV",
            data=csv_buf.getvalue(),
            file_name="rail_shortest_paths.csv",
            mime="text/csv",
        )

else:
    st.info("Upload both the FRA segments CSV and the ZIP OD pairs CSV to begin.")
