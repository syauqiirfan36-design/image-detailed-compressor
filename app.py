import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage import color
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="PCA Image Compression",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS ringan untuk membuat UI lebih clean & modern
st.markdown("""
<style>
    /* Mengubah jenis font global */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Bayangan pada gambar dan card ringan */
    .stImage > img {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #f1f1f1;
    }
    
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e9ecef;
        margin-bottom: 10px;
    }
    
    .metric-box span {
        font-size: 13px;
        color: #6c757d;
        display: block;
        margin-bottom: 5px;
    }
    
    .metric-box strong {
        font-size: 15px;
        color: #212529;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #1f2937;
        font-weight: 700 !important;
    }
    
    hr {
        margin: 2em 0;
        border-color: #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)
st.title("📷 PCA Image Compression Analysis")
st.markdown("Aplikasi web **Full Python** untuk mengevaluasi kompresi citra menggunakan *Principal Component Analysis* (PCA).")
# Sidebar untuk Input Parameter
st.sidebar.header("⚙️ Pengaturan")
uploaded_file = st.sidebar.file_uploader("Unggah Citra (JPG/PNG)", type=["jpg", "jpeg", "png"])
k_input = st.sidebar.text_input("Nilai Komponen (k)", value="5, 10, 50, 100", help="Masukkan angka yang dipisahkan dengan koma.")
if uploaded_file is not None:
    try:
        # 1. Load citra
        img = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(img)
        
        # 2. Konversi ke Grayscale (Jika RGB)
        if img_array.ndim == 3:
            img_gray = color.rgb2gray(img_array) * 255
        else:
            img_gray = img_array.astype(float)
            
        X = img_gray
        H, W = X.shape
        # 3. Parse list nilai k
        k_values = sorted(list(set([int(k.strip()) for k in k_input.split(',') if k.strip().isdigit()])))
        
        if not k_values:
            st.sidebar.error("Masukkan nilai k yang valid!")
            st.stop()
        # Tombol Mulai Analisis
        if st.sidebar.button("🚀 Mulai Analisis", type="primary", use_container_width=True):
            
            # --- BAGIAN 1: EDA AWAL ---
            st.markdown("---")
            st.header("1. EDA Awal (Sebelum PCA)")
            
            pix_mean = X.mean()
            pix_std = X.std()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(X, caption=f"Citra Asli (Grayscale)\nDimensi: {H}x{W} px", use_column_width=True, clamp=True)
                st.info(f"**Mean Intensitas:** {pix_mean:.2f}\n\n**Standard Deviation:** {pix_std:.2f}\n\n**Total Piksel:** {H*W:,}")
                
            with col2:
                fig_hist_orig, ax_hist = plt.subplots(figsize=(7, 4.5))
                ax_hist.hist(X.flatten(), bins=64, color='steelblue', edgecolor='white', alpha=0.9)
                ax_hist.set_title("Histogram Distribusi Intensitas Piksel", fontweight='bold')
                ax_hist.axvline(pix_mean, color='red', linestyle='--', linewidth=2, label=f'Mean = {pix_mean:.1f}')
                ax_hist.legend()
                ax_hist.grid(axis='y', alpha=0.3)
                ax_hist.spines['top'].set_visible(False)
                ax_hist.spines['right'].set_visible(False)
                st.pyplot(fig_hist_orig)
            # --- BAGIAN 2: PCA EDA ---
            st.markdown("---")
            st.header("2. EDA Saat PCA (Pemilihan K)")
            
            with st.spinner("Menghitung model PCA penuh..."):
                pca_full = PCA()
                pca_full.fit(X)
                eigenvalues = pca_full.explained_variance_
                cum_var = np.cumsum(pca_full.explained_variance_ratio_)
                n_components_total = len(eigenvalues)
                
            col3, col4 = st.columns(2)
            
            with col3:
                fig_scree, ax_scree = plt.subplots(figsize=(6, 4))
                n_show = min(50, n_components_total)
                ax_scree.plot(range(1, n_show + 1), eigenvalues[:n_show], 'o-', color='royalblue', markersize=5, linewidth=2)
                ax_scree.set_title(f"Scree Plot (50 Komponen Pertama)", fontweight='bold')
                ax_scree.set_xlabel("Jumlah Komponen (k)")
                ax_scree.set_ylabel("Eigenvalue")
                ax_scree.grid(alpha=0.3)
                ax_scree.spines['top'].set_visible(False)
                ax_scree.spines['right'].set_visible(False)
                st.pyplot(fig_scree)
                
            with col4:
                fig_cum, ax_cum = plt.subplots(figsize=(6, 4))
                ax_cum.plot(range(1, n_components_total + 1), cum_var * 100, color='darkorange', linewidth=2.5)
                ax_cum.axhline(90, color='blue', linestyle='--', alpha=0.6, label='90% Var')
                ax_cum.axhline(95, color='red', linestyle=':', alpha=0.6, label='95% Var')
                ax_cum.set_title("Cumulative Explained Variance (%)", fontweight='bold')
                ax_cum.set_xlabel("Jumlah Komponen (k)")
                ax_cum.set_ylabel("Variansi Kumulatif (%)")
                ax_cum.legend()
                ax_cum.grid(alpha=0.3)
                ax_cum.spines['top'].set_visible(False)
                ax_cum.spines['right'].set_visible(False)
                st.pyplot(fig_cum)
            # Validasi K values vs Dimensi
            k_values = [k for k in k_values if k <= n_components_total]
            if not k_values:
                st.error(f"Semua nilai K yang dimasukkan lebih besar dari dimensi maksimal gambar ({n_components_total}).")
                st.stop()
            # --- BAGIAN 3: REKONSTRUKSI CITRA (GRID) ---
            st.markdown("---")
            st.header("3. Hasil Rekonstruksi & Error Image")
            st.markdown("Perbandingan kualitas citra yang telah dikompresi menggunakan berbagai jumlah komponen (k). **Maksimal 4 gambar per baris.**")
            
            results = []
            metric_data = {'k': [], 'ev': [], 'mse': [], 'psnr': [], 'ssim': [], 'cr': []}
            
            with st.spinner("Memproses rekonstruksi gambar untuk tiap nilai K..."):
                for k in k_values:
                    pca_k = PCA(n_components=k)
                    X_comp = pca_k.fit_transform(X)
                    X_rec = pca_k.inverse_transform(X_comp)
                    X_rec = np.clip(X_rec, 0, 255)
                    
                    # Kalkulasi Metrik
                    mse = np.mean((X - X_rec) ** 2)
                    psnr = 10 * np.log10((255.0 ** 2) / mse) if mse > 0 else float('inf')
                    ssim_val = ssim(X, X_rec, data_range=255)
                    cr = (H * W) / (k * (H + W + 1))
                    ev = sum(pca_full.explained_variance_ratio_[:k]) * 100
                    
                    metric_data['k'].append(k)
                    metric_data['ev'].append(ev)
                    metric_data['mse'].append(mse)
                    metric_data['psnr'].append(psnr)
                    metric_data['ssim'].append(ssim_val)
                    metric_data['cr'].append(cr)
                    
                    error_img = np.abs(X - X_rec)
                    
                    results.append({
                        'k': k, 'rec': X_rec, 'err': error_img,
                        'ev': ev, 'cr': cr, 'psnr': psnr, 'ssim': ssim_val
                    })
            # Menampilkan Grid Fleksibel: max 4 columns
            max_cols = 4
            for i in range(0, len(results), max_cols):
                cols = st.columns(max_cols)
                batch = results[i:i+max_cols]
                
                for col_idx, res in enumerate(batch):
                    with cols[col_idx]:
                        st.markdown(f"<h3 style='text-align:center;'>K = {res['k']}</h3>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class='metric-box'>
                            <span>Explained Var: <strong>{res['ev']:.1f}%</strong></span>
                            <span>Compression Ratio: <strong>{res['cr']:.1f}x</strong></span>
                            <span>PSNR: <strong>{res['psnr']:.1f} dB</strong></span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.image(res['rec'], caption="1. Hasil Rekonstruksi", use_column_width=True, clamp=True)
                        
                        fig_h, ax_h = plt.subplots(figsize=(3, 2.5))
                        ax_h.hist(X.flatten(), bins=64, color='steelblue', alpha=0.5, density=True)
                        ax_h.hist(res['rec'].flatten(), bins=64, color='darkorange', alpha=0.7, density=True)
                        ax_h.axis('off')
                        plt.tight_layout(pad=0)
                        st.pyplot(fig_h, clear_figure=True)
                        st.caption("2. Histogram Komparasi")
                        
                        st.image(res['err'], caption="3. Error Image (Selisih Absolut)", use_column_width=True, clamp=True)
                        st.markdown("<br>", unsafe_allow_html=True)
            # --- BAGIAN 4: METRIK EVALUASI KESELURUHAN ---
            st.markdown("---")
            st.header("4. Evaluasi Metrik Kompresi")
            
            with st.expander("ℹ️ Penjelasan Singkat Metrik (Klik untuk buka)", expanded=True):
                st.markdown("""
                * **Explained Var (%)**: Persentase informasi asli yang dipertahankan. (Mendekati 100% lebih baik)
                * **MSE (Mean Squared Error)**: Rata-rata error piksel citra asli dengan citra rekonstruksi. (Mendekati 0 lebih baik)
                * **PSNR (Peak Signal-to-Noise Ratio)**: Kualitas visual berdasarkan level *noise* error. (> 30 dB = Baik)
                * **SSIM (Structural Similarity)**: Kemiripan struktur dan bentuk tepi. (Mendekati 1 lebih baik)
                * **Compression Ratio**: Perbandingan ukuran data asli dan data terkompresi. (Lebih tinggi = file lebih kecil / memori lebih hemat)
                """)
            if len(k_values) > 1:
                x_pos = np.arange(len(k_values))
                k_labels = [str(k) for k in k_values]
                
                fig_m, axes_m = plt.subplots(2, 3, figsize=(16, 9))
                axes_m = axes_m.flatten()
                
                # Setup Global Title
                # fig_m.suptitle("Grafik Perbandingan Metrik vs Nilai K", fontsize=16, fontweight='bold', y=1.02)
                
                # Fungsi Helper untuk plot Seragam
                def plot_bar_line(ax, data_list, title, ylabel, color_bar, color_line, marker):
                    ax.bar(x_pos, data_list, color=color_bar, alpha=0.6)
                    ax.plot(x_pos, data_list, color=color_line, marker=marker, linestyle='-', linewidth=2.5, markersize=8)
                    ax.set_title(title, fontweight='bold', pad=15)
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(k_labels)
                    ax.set_ylabel(ylabel)
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                plot_bar_line(axes_m[0], metric_data['ev'], 'Explained Variance (%)', '% Variansi', 'lightblue', 'blue', 'o')
                plot_bar_line(axes_m[1], metric_data['mse'], 'Mean Squared Error (MSE)', 'Error (Lower is Better)', 'lightcoral', 'red', 's')
                
                # Khusus PSNR (Ada Garis Ambang Batas 30dB)
                axes_m[2].bar(x_pos, metric_data['psnr'], color='lightgreen', alpha=0.6)
                axes_m[2].plot(x_pos, metric_data['psnr'], color='green', marker='^', linestyle='-', linewidth=2.5, markersize=8)
                axes_m[2].axhline(30, color='red', linestyle='--', linewidth=2, label='Standar Baik (30 dB)')
                axes_m[2].set_title('Peak Signal-to-Noise Ratio (PSNR)', fontweight='bold', pad=15)
                axes_m[2].set_xticks(x_pos)
                axes_m[2].set_xticklabels(k_labels)
                axes_m[2].set_ylabel('dB (Higher is Better)')
                axes_m[2].grid(True, alpha=0.3, axis='y')
                axes_m[2].spines['top'].set_visible(False)
                axes_m[2].spines['right'].set_visible(False)
                axes_m[2].legend()
                
                plot_bar_line(axes_m[3], metric_data['ssim'], 'Structural Similarity Index (SSIM)', 'SSIM (Mendekati 1 is Better)', 'thistle', 'purple', 'd')
                plot_bar_line(axes_m[4], metric_data['cr'], 'Compression Ratio', 'CR (Higher is Better)', 'navajowhite', 'darkorange', 'x')
                
                axes_m[5].axis('off') # Kosongkan slot ke-6
                
                plt.tight_layout(pad=3.0)
                st.pyplot(fig_m)
                
            else:
                st.warning("📊 Untuk melihat grafik perbandingan metrik evaluasi, silakan masukkan **minimal 2 nilai K yang berbeda** pada menu di samping kiri.")
                
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses gambar: {str(e)}")
else:
    st.info("👈 Silakan unggah gambar pada **Panel Pengaturan (Sidebar)** di sebelah kiri untuk memulai analisis PCA.")