import os
import gradio as gr
from src.searcher import Searcher
from src.indexer import Indexer
from src.grapher import Grapher
from src.graph_component import generate_html_plot
from src.settings import Settings

settings = Settings()

def format_gallery_results(paths, all_paths):
    results = []
    for path in paths:
        try:
            idx = all_paths.index(path)
        except ValueError:
            idx = "?"
        filename = os.path.basename(path)
        caption = f"{filename} | {idx} "
        results.append((path, caption))
    return results

def perform_search(text_query, image_query, top_k):
    top_k = int(top_k)
    if image_query is not None:
        paths = search_backend.search(image_query, top_k)
    else:
        paths = search_backend.search(text_query, top_k)
        
    # Format the gallery tuples
    gallery_data = format_gallery_results(paths, search_backend.image_paths)
    
    total_count = f"<div style='text-align: right; font-size: 0.9em; color: #6b7280; padding-top: 4px; padding-right: 8px;'>Model: {settings.current_model_id} | Total Images: {len(search_backend.image_paths)}</div>"
    
    return gallery_data, total_count

def rebuild_index():
    index_backend.build_Index()
    search_backend.reload_index()
    
    # Show all images after a rebuild
    paths = search_backend.image_paths
    gallery_data = format_gallery_results(paths, paths)
    
    total_count = f"<div style='text-align: right; font-size: 0.9em; color: #6b7280; padding-top: 4px; padding-right: 8px;'>Model: {settings.current_model_id} | Total Images: {len(paths)}</div>"
    
    return gallery_data, total_count

def update_latent_plot(x_text, y_text, offset):
    df = graph_backend.generate_plot_data(x_text, y_text, offset)
    if df.empty:
        return "<div style='text-align:center; padding:50px;'>Not enough data to plot or offset out of bounds.</div>"

    html_plot = generate_html_plot(df, x_text, y_text)
    return html_plot

def change_model_and_rebuild(new_model_id):
    global index_backend, search_backend, graph_backend
    
    if not new_model_id or not new_model_id.strip():
        new_model_id = "openai/clip-vit-base-patch32"
        
    new_model_id = new_model_id.strip()
    print(f"Switching model to: {new_model_id}")
    
    # Use the class to save the setting and initialize the new hardware/model
    settings.save_settings(new_model_id)
    device, model, processor = settings.initialize_backend(new_model_id)
    
    index_backend = Indexer(device, model, processor)
    search_backend = Searcher(device, model, processor)
    graph_backend = Grapher(device, model, processor, search_backend)
    
    index_backend.build_Index()
    search_backend.reload_index()
    
    paths = search_backend.image_paths
    gallery_data = format_gallery_results(paths, paths)
    
    total_count = f"<div style='text-align: right; font-size: 0.9em; color: #6b7280; padding-top: 4px; padding-right: 8px;'>Model: {settings.current_model_id} | Total Images: {len(paths)}</div>"
    
    return gallery_data, total_count

# -------------------------------------------------------------------
# Gradio Interface
# -------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(), title="Local Image Search", fill_height=True) as app:
    gr.Markdown("# Local Semantic Image Search")

# -------------------------------------------------------------------
# Search
# -------------------------------------------------------------------    
    with gr.Tabs():
        with gr.Tab("Semantic Search"):
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    search_query = gr.Textbox(
                        label="Search by Text", 
                        placeholder="Leave blank to see all images, or type a query...",
                        max_lines=1
                    )
                    image_query = gr.Image(
                        label="Search by Image",
                        type="pil",
                        height=250,
                        sources=["upload"]
                    )
                    with gr.Group():
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=3,
                            step=1, 
                            label="Max Results"
                        )
                        search_btn = gr.Button("Search", variant="primary")
                    rebuild_btn = gr.Button("Rebuild Index", variant="secondary")
                
                with gr.Column(scale=3):
                    results_gallery = gr.Gallery(
                        label="Gallery", 
                        show_label=False, 
                        columns=4, 
                        rows=4,
                        object_fit="scale-down", 
                        height="85vh", 
                        interactive=False
                    )
                    total_count_display = gr.HTML()

            # Search
            search_btn.click(fn=perform_search, inputs=[search_query, image_query, top_k_slider], outputs=[results_gallery, total_count_display])
            search_query.submit(fn=perform_search, inputs=[search_query, image_query, top_k_slider], outputs=[results_gallery, total_count_display])
            
            # Rebuild
            rebuild_btn.click(fn=rebuild_index, inputs=[], outputs=[results_gallery, total_count_display])

            # Initial load sends blank search to show all images
            app.load(fn=perform_search, inputs=[search_query, image_query, top_k_slider], outputs=[results_gallery, total_count_display]) 

# -------------------------------------------------------------------
# Graph
# -------------------------------------------------------------------
        with gr.Tab("Latent Space Graph"):
            gr.Markdown("Map your images across two distinct semantic concepts (Max 500 images per batch).")
            with gr.Row():
                x_axis_input = gr.Textbox(label="X-Axis",placeholder="Nature", scale=2)
                y_axis_input = gr.Textbox(label="Y-Axis",placeholder="Industrial", scale=2)
                offset_input = gr.Number(label="Index Offset", value=0, precision=0, scale=1)
                plot_btn = gr.Button("Map Latent Space", variant="primary", scale=1)
            
            with gr.Row():
                latent_plot = gr.HTML(label="Latent Space")
                
            plot_btn.click(fn=update_latent_plot, inputs=[x_axis_input, y_axis_input, offset_input], outputs=latent_plot)
            x_axis_input.submit(fn=update_latent_plot, inputs=[x_axis_input, y_axis_input, offset_input], outputs=latent_plot)
            y_axis_input.submit(fn=update_latent_plot, inputs=[x_axis_input, y_axis_input, offset_input], outputs=latent_plot)
            offset_input.submit(fn=update_latent_plot, inputs=[x_axis_input, y_axis_input, offset_input], outputs=latent_plot)
            
# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
        with gr.Tab("Settings"):
            gr.Markdown("### Application Settings")
            with gr.Row():
                with gr.Column(scale=2):
                    model_input = gr.Textbox(
                        label="Hugging Face Model ID", 
                        value=settings.current_model_id, # Pull from the class on boot
                        info="Warning: Changing the model will completely overwrite your current FAISS database. It may take a moment to download new weights and re-embed all images."
                    )
                with gr.Column(scale=1):
                    apply_btn = gr.Button("Apply & Rebuild Index", variant="primary")
                    status_text = gr.Markdown("") 
                    
            apply_btn.click(
                fn=lambda: "**Status:** Loading new model and rebuilding database... Please wait.", 
                outputs=[status_text]
            ).then(
                fn=change_model_and_rebuild, 
                inputs=[model_input], 
                outputs=[results_gallery, total_count_display]
            ).then(
                fn=lambda: "**Status:** Success! New model loaded and index rebuilt.", 
                outputs=[status_text]
            )

if __name__ == "__main__":
    shared_device, shared_model, shared_processor = settings.initialize_backend()
    
    index_backend = Indexer(shared_device, shared_model, shared_processor)

    if not os.path.exists("./data/embeddings.faiss"):
        print("Database not found. Building index for the first time...")
        os.makedirs("./data", exist_ok=True)
        index_backend.build_Index()
    else:
        print("Database found. Skipping initial build.")

    search_backend = Searcher(shared_device, shared_model, shared_processor)
    graph_backend = Grapher(shared_device, shared_model, shared_processor, search_backend)

    app.launch(server_name="127.0.0.1", server_port=7860, share=True)