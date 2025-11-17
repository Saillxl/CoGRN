import os

def main():
    from src.utils import get_TFBS_from_ATAC, get_TFBS_from_promoter, build_Graph, build_cross_graph, extract_atac_feature, extract_scrna_feature2

    atac_file = "./dataset/H_kidney/ATAC_matrix/matrix.mtx"
    scrna_file = "./dataset/H_kidney/RNA_matrix/matrix.mtx"

    save_dir = "./result/kidney"

    motif_dir = "./data_resource/HOCOMOCOv11"
    genome_file = "./data_resource/hg38.fa"
    promoter_file = "./data_resource/gencode.v38.ProteinCoding_gene_promoter.txt"

    atac_region_file = get_TFBS_from_ATAC(atac_file, motif_dir, genome_file, overwrite=False, save_dir=save_dir)
    print("atac_region_file:",atac_region_file)

    promoter_region_file = get_TFBS_from_promoter(promoter_file, motif_dir, overwrite=False, save_dir=save_dir)
    print(promoter_region_file)

    graph_file, node_file = build_Graph(atac_region_file, promoter_region_file, overwrite=False, save_dir=save_dir)
    print("node_file:",node_file)

    atac_feature_file = extract_atac_feature(atac_file, node_file, cell_num=None, seed=666, overwrite=False, save_dir=save_dir)
    print("atac_teature_file:",atac_feature_file)

    scrna_feature_file = extract_scrna_feature2(scrna_file, node_file, cell_num=None, seed=666, overwrite=False, just_node=False, save_dir=save_dir)
    print(scrna_feature_file)


    cross_graph_file, cross_node_file, cross_atac_file, aross_scrna_file = \
        build_cross_graph(graph_file, atac_feature_file, scrna_feature_file, overwrite=True, save_dir=save_dir)
    print(cross_graph_file)



if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)  # 可选但推荐
    main()
