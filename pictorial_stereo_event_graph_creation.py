from graph_creation.dsec_graph_creation import stereo_event_graph_creation_with_gui

if __name__ == "__main__":
    stereo_event_graph_creation_with_gui(
        time_window_us=5000,
        how_many_right_events_per_draw = 10,
        # sequence_folder_path="/media/enrico/Enrico_Datasets/dsec/train/zurich_city_09_a",
        neighbors=10
    )