import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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

st.markdown("""
<style>
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
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
    .metric-box span { font-size: 13px; color: #6c757d; display: block; margin-bottom: 5px; }
    .metric-box strong { font-size: 15px; color: #212529; }
    h1, h2, h3 { color: #1f2937; font-weight: 700 !important; }
    hr { margin: 2em 0; border-color: #e5e7eb; }
</style>
""", unsafe_allow_html=True)

st.title("📷 PCA Image Compression Analysis")
st.markdown("Aplikasi web **Full Python** untuk mengevaluasi kompresi citra menggunakan *Principal Component Analysis* (PCA).")

st.sidebar.header("⚙️ Pengaturan")
uploaded_file = st.sidebar.file_uploader("Unggah Citra (JPG/PNG)", type=["jpg", "jpeg", "png"])
k_input = st.sidebar.text_input("Nilai Komponen (k)", value="5, 10, 50, 100",
                                  help="Masukkan angka yang dipisahkan dengan koma.")

# ── Helper: tampilkan gambar grayscale dengan benar ──────────────────────────
# FIX UTAMA: st.image() dengan numpy 2D float bisa salah interpret (jadi putih).
# Solusi: konversi eksplisit ke PIL Image mode 'L' sebelum ditampilkan.
def show_gray(arr, caption="", width=None):
    pil_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode='L')
    st.image(pil_img, caption=caption, use_column_width=(width is None), width=width)


# ── Cache PCA penuh agar tidak re-fit setiap interaksi UI ───────────────────
@st.cache_data(show_spinner=False)
def fit_full_pca(X_bytes, shape):
    """Cache PCA penuh — ini yang paling berat (O(min(H,W)^2 * max(H,W)))."""
    X = np.frombuffer(X_bytes, dtype=np.float64).reshape(shape)
    pca_full = PCA()
    pca_full.fit(X)
    return pca_full.explained_variance_, pca_full.explained_variance_ratio_


if uploaded_file is not None:
    try:
        # ── Load & resize citra ──────────────────────────────────────────────
        img = Image.open(uploaded_file).convert('L')   # langsung Grayscale
        MAX_DIM = 800
        if max(img.size) > MAX_DIM:
            img.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.LANCZOS)

        img_array = np.array(img)                      # uint8, 0-255
        X = img_array.astype(np.float64)               # float untuk PCA
        H, W = X.shape

        # ── Parse nilai k ───────────────────────────────────────────────────
        k_values = sorted(list(set(
            [int(k.strip()) for k in k_input.split(',') if k.strip().isdigit()]
        )))
        if not k_values:
            st.sidebar.error("Masukkan nilai k yang valid!")
            st.stop()

        if st.sidebar.button("🚀 Mulai Analisis", type="primary", use_container_width=True):

            # ════════════════════════════════════════════════════════════════
            # BAGIAN 1 — EDA AWAL (SEBELUM PCA)
            # ════════════════════════════════════════════════════════════════
            st.markdown("---")
            st.header("1. EDA Awal (Sebelum PCA)")

            pix_min  = X.min()
            pix_max  = X.max()
            pix_mean = X.mean()
            pix_std  = X.std()

            col1, col2 = st.columns([1, 2])

            with col1:
                # ── FIX: gunakan show_gray() ──────────────────────────────
                show_gray(X, caption=f"Citra Asli (Grayscale)\nDimensi: {H}×{W} px")

                # [1.1] Ukuran & statistik
                st.info(
                    f"**Dimensi:** {H} × {W} px ({H*W:,} piksel)  \n"
                    f"**Min / Max:** {pix_min:.0f} / {pix_max:.0f}  \n"
                    f"**Mean Intensitas:** {pix_mean:.2f}  \n"
                    f"**Standard Deviation:** {pix_std:.2f}"
                )

                # [1.2] Interpretasi mean
                if pix_mean < 85:
                    mean_interp = f"Mean = **{pix_mean:.1f}** → Citra cenderung **GELAP** (dominan hitam/abu gelap)."
                elif pix_mean > 170:
                    mean_interp = f"Mean = **{pix_mean:.1f}** → Citra cenderung **TERANG** (dominan putih/abu terang)."
                else:
                    mean_interp = f"Mean = **{pix_mean:.1f}** → Kecerahan **SEDANG**, distribusi relatif merata."

                # [1.3] Interpretasi std
                if pix_std < 40:
                    std_interp = f"Std = **{pix_std:.1f}** → Kontras **RENDAH**, piksel cenderung seragam."
                elif pix_std > 80:
                    std_interp = f"Std = **{pix_std:.1f}** → Kontras **TINGGI**, variasi gelap-terang signifikan."
                else:
                    std_interp = f"Std = **{pix_std:.1f}** → Kontras **SEDANG**, variasi intensitas normal."

                st.markdown(f"🔍 {mean_interp}  \n🔍 {std_interp}")

            with col2:
                fig_hist, ax_hist = plt.subplots(figsize=(7, 4.5))
                ax_hist.hist(X.flatten(), bins=64, color='steelblue', edgecolor='white', alpha=0.9)
                ax_hist.set_title("Histogram Distribusi Intensitas Piksel", fontweight='bold')
                ax_hist.set_xlabel("Nilai Intensitas (0=Hitam, 255=Putih)")
                ax_hist.set_ylabel("Frekuensi")
                ax_hist.axvline(pix_mean, color='red', linestyle='--', linewidth=2,
                                label=f'Mean = {pix_mean:.1f}')
                ax_hist.legend()
                ax_hist.grid(axis='y', alpha=0.3)
                ax_hist.spines['top'].set_visible(False)
                ax_hist.spines['right'].set_visible(False)
                st.pyplot(fig_hist)
                plt.close(fig_hist)

                # [1.4] Interpretasi histogram
                hist_vals, bin_edges = np.histogram(X.flatten(), bins=64, range=(0, 255))
                low_ratio  = hist_vals[:10].sum()  / hist_vals.sum()
                high_ratio = hist_vals[-10:].sum() / hist_vals.sum()
                active_idx = np.where(hist_vals > hist_vals.max() * 0.05)[0]
                spread = bin_edges[active_idx[-1]] - bin_edges[active_idx[0]] if len(active_idx) > 1 else 0

                if low_ratio > 0.4:
                    hist_msg = f"Banyak piksel dekat 0 ({low_ratio*100:.1f}%) → citra **GELAP**."
                elif high_ratio > 0.4:
                    hist_msg = f"Banyak piksel dekat 255 ({high_ratio*100:.1f}%) → citra **TERANG**."
                else:
                    hist_msg = "Distribusi intensitas tersebar → variasi kecerahan yang baik."

                if spread > 180:
                    spread_msg = f"Sebaran lebar (±{spread:.0f}) → **kontras TINGGI**."
                elif spread < 80:
                    spread_msg = f"Sebaran sempit (±{spread:.0f}) → **kontras RENDAH**, citra cenderung monoton."
                else:
                    spread_msg = f"Sebaran sedang (±{spread:.0f}) → kontras normal."

                st.caption(f"📊 {hist_msg} {spread_msg}")

            # ════════════════════════════════════════════════════════════════
            # BAGIAN 2 — EDA SAAT PCA (PEMILIHAN K)
            # ════════════════════════════════════════════════════════════════
            st.markdown("---")
            st.header("2. EDA Saat PCA (Pemilihan K)")

            with st.spinner("Menghitung PCA penuh... (di-cache untuk efisiensi)"):
                # Serialize X sebagai bytes agar bisa di-cache oleh st.cache_data
                eigenvalues, explained_var_ratio = fit_full_pca(
                    X.tobytes(), X.shape
                )
                cum_var = np.cumsum(explained_var_ratio)
                n_components_total = len(eigenvalues)

            # Threshold info
            thresh_info = {}
            for t in [0.80, 0.90, 0.95, 0.99]:
                thresh_info[t] = int(np.argmax(cum_var >= t) + 1)

            # Elbow detection
            diffs = np.diff(eigenvalues)
            elbow_idx = int(np.argmin(diffs) + 1)

            # Info ringkas
            col_t1, col_t2, col_t3, col_t4 = st.columns(4)
            col_t1.metric("k untuk 80% var", thresh_info[0.80])
            col_t2.metric("k untuk 90% var", thresh_info[0.90])
            col_t3.metric("k untuk 95% var", thresh_info[0.95])
            col_t4.metric("k untuk 99% var", thresh_info[0.99])

            col3, col4 = st.columns(2)

            with col3:
                n_show = min(50, n_components_total)
                fig_scree, ax_scree = plt.subplots(figsize=(6, 4))
                ax_scree.plot(range(1, n_show + 1), eigenvalues[:n_show],
                              'o-', color='royalblue', markersize=5, linewidth=2)
                ax_scree.axvline(elbow_idx, color='red', linestyle='--', linewidth=1.5,
                                 label=f'Elbow ≈ k={elbow_idx}')
                ax_scree.set_title(f"Scree Plot (50 Komponen Pertama)", fontweight='bold')
                ax_scree.set_xlabel("Jumlah Komponen (k)")
                ax_scree.set_ylabel("Eigenvalue")
                ax_scree.legend()
                ax_scree.grid(alpha=0.3)
                ax_scree.spines['top'].set_visible(False)
                ax_scree.spines['right'].set_visible(False)
                st.pyplot(fig_scree)
                plt.close(fig_scree)
                st.caption(
                    f"🔍 Komponen 1–{elbow_idx}: membawa informasi BESAR (eigenvalue tinggi).  \n"
                    f"Setelah k≈{elbow_idx}: kurva melandai → informasi kecil. "
                    f"Titik elbow = sweet spot kompresi vs. kualitas."
                )

            with col4:
                fig_cum, ax_cum = plt.subplots(figsize=(6, 4))
                ax_cum.plot(range(1, n_components_total + 1), cum_var * 100,
                            color='darkorange', linewidth=2.5)
                for t, color_, ls, label in [
                    (0.80, 'green', '--', f'80% → k={thresh_info[0.80]}'),
                    (0.90, 'blue',  '-.', f'90% → k={thresh_info[0.90]}'),
                    (0.95, 'red',   ':',  f'95% → k={thresh_info[0.95]}'),
                    (0.99, 'purple','-',  f'99% → k={thresh_info[0.99]}'),
                ]:
                    ax_cum.axhline(t * 100, color=color_, linestyle=ls,
                                   linewidth=1.2, label=label)
                ax_cum.set_title("Cumulative Explained Variance (%)", fontweight='bold')
                ax_cum.set_xlabel("Jumlah Komponen (k)")
                ax_cum.set_ylabel("Variansi Kumulatif (%)")
                ax_cum.legend(fontsize=8)
                ax_cum.grid(alpha=0.3)
                ax_cum.spines['top'].set_visible(False)
                ax_cum.spines['right'].set_visible(False)
                st.pyplot(fig_cum)
                plt.close(fig_cum)
                st.caption(
                    "🔍 Semakin banyak komponen → lebih banyak informasi dipertahankan, "
                    "tapi rasio kompresi mengecil. Ini adalah **trade-off utama PCA**."
                )

            # Validasi K vs dimensi
            k_values = [k for k in k_values if k <= n_components_total]
            if not k_values:
                st.error(f"Semua nilai K melebihi dimensi maksimal gambar ({n_components_total}).")
                st.stop()

            # ════════════════════════════════════════════════════════════════
            # BAGIAN 3 — REKONSTRUKSI & ERROR IMAGE
            # ════════════════════════════════════════════════════════════════
            st.markdown("---")
            st.header("3. Hasil Rekonstruksi & Error Image")
            st.markdown("Perbandingan kualitas citra terkompresi per nilai k. **Maks. 4 gambar per baris.**")

            results = []
            metric_data = {'k': [], 'ev': [], 'mse': [], 'psnr': [], 'ssim': [], 'cr': []}

            with st.spinner("Memproses rekonstruksi untuk tiap nilai K..."):
                for k in k_values:
                    pca_k   = PCA(n_components=k)
                    X_comp  = pca_k.fit_transform(X)
                    X_rec   = np.clip(pca_k.inverse_transform(X_comp), 0, 255)

                    mse_val  = np.mean((X - X_rec) ** 2)
                    psnr_val = 10 * np.log10((255.0 ** 2) / mse_val) if mse_val > 0 else float('inf')
                    ssim_val = ssim(X, X_rec, data_range=255)
                    cr_val   = (H * W) / (k * (H + W + 1))
                    ev_val   = float(np.sum(explained_var_ratio[:k]) * 100)

                    for key, val in zip(['k','ev','mse','psnr','ssim','cr'],
                                        [k, ev_val, mse_val, psnr_val, ssim_val, cr_val]):
                        metric_data[key].append(val)

                    results.append({
                        'k': k, 'rec': X_rec, 'err': np.abs(X - X_rec),
                        'ev': ev_val, 'cr': cr_val, 'psnr': psnr_val,
                        'ssim': ssim_val, 'mse': mse_val
                    })

            max_cols = 4
            for i in range(0, len(results), max_cols):
                cols  = st.columns(max_cols)
                batch = results[i:i + max_cols]

                for col_idx, res in enumerate(batch):
                    with cols[col_idx]:
                        st.markdown(f"<h3 style='text-align:center;'>K = {res['k']}</h3>",
                                    unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class='metric-box'>
                            <span>Explained Var: <strong>{res['ev']:.1f}%</strong></span>
                            <span>Compression Ratio: <strong>{res['cr']:.1f}x</strong></span>
                            <span>PSNR: <strong>{res['psnr']:.1f} dB</strong></span>
                            <span>SSIM: <strong>{res['ssim']:.4f}</strong></span>
                        </div>
                        """, unsafe_allow_html=True)

                        # ── FIX: gunakan show_gray() ──────────────────────
                        show_gray(res['rec'], caption="1. Hasil Rekonstruksi")

                        fig_h, ax_h = plt.subplots(figsize=(3, 2.5))
                        ax_h.hist(X.flatten(), bins=64, color='steelblue', alpha=0.5, density=True)
                        ax_h.hist(res['rec'].flatten(), bins=64, color='darkorange', alpha=0.7, density=True)
                        ax_h.axis('off')
                        plt.tight_layout(pad=0)
                        st.pyplot(fig_h, clear_figure=True)
                        plt.close(fig_h)
                        st.caption("2. Histogram: Asli (biru) vs Rekonstruksi (oranye)")

                        # Error image dengan colormap 'hot'
                        error_norm = np.clip(res['err'] / 50.0, 0, 1)
                        error_rgb  = (plt.cm.hot(error_norm)[..., :3] * 255).astype(np.uint8)
                        st.image(Image.fromarray(error_rgb), caption="3. Error Image (Terang = Error Besar)",
                                 use_column_width=True)

                        # Interpretasi singkat per K
                        if res['psnr'] > 40:
                            q_label = "✅ Sangat Baik"
                        elif res['psnr'] > 30:
                            q_label = "🟢 Baik"
                        elif res['psnr'] > 20:
                            q_label = "🟡 Cukup"
                        else:
                            q_label = "🔴 Buruk"
                        st.caption(f"Kualitas: **{q_label}**")
                        st.markdown("<br>", unsafe_allow_html=True)

            # ════════════════════════════════════════════════════════════════
            # BAGIAN 4 — TABEL & GRAFIK METRIK EVALUASI
            # ════════════════════════════════════════════════════════════════
            st.markdown("---")
            st.header("4. Evaluasi Metrik Kompresi")

            with st.expander("ℹ️ Penjelasan Singkat Metrik", expanded=True):
                st.markdown("""
                | Metrik | Keterangan | Ideal |
                |---|---|---|
                | **Explained Var (%)** | Informasi asli yang dipertahankan | Mendekati 100% |
                | **MSE** | Rata-rata error piksel² | Mendekati 0 |
                | **PSNR (dB)** | Kualitas visual berdasarkan noise | > 30 dB = Baik |
                | **SSIM** | Kemiripan struktur & tepi | Mendekati 1 |
                | **Compression Ratio** | Ukuran asli ÷ ukuran terkompresi | Lebih tinggi = lebih hemat |
                """)

            # ── Tabel ringkas (seperti Colab) ────────────────────────────────
            st.subheader("📋 Tabel Perbandingan Metrik")
            import pandas as pd
            df_metrics = pd.DataFrame({
                'k': metric_data['k'],
                'Expl. Var (%)': [f"{v:.2f}" for v in metric_data['ev']],
                'MSE':           [f"{v:.2f}" for v in metric_data['mse']],
                'PSNR (dB)':     [f"{v:.2f}" for v in metric_data['psnr']],
                'SSIM':          [f"{v:.4f}" for v in metric_data['ssim']],
                'Comp. Ratio':   [f"{v:.2f}x" for v in metric_data['cr']],
                'Kualitas':      [
                    "Sangat Baik" if p > 40 else "Baik" if p > 30 else "Cukup" if p > 20 else "Buruk"
                    for p in metric_data['psnr']
                ]
            })
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

            # ── Grafik evaluasi ──────────────────────────────────────────────
            if len(k_values) > 1:
                x_pos   = np.arange(len(k_values))
                k_labels = [str(k) for k in k_values]

                fig_m, axes_m = plt.subplots(2, 3, figsize=(16, 9))
                axes_m = axes_m.flatten()

                def plot_bar_line(ax, data_list, title, ylabel, color_bar, color_line, marker):
                    ax.bar(x_pos, data_list, color=color_bar, alpha=0.6)
                    ax.plot(x_pos, data_list, color=color_line, marker=marker,
                            linestyle='-', linewidth=2.5, markersize=8)
                    ax.set_title(title, fontweight='bold', pad=15)
                    ax.set_xticks(x_pos); ax.set_xticklabels(k_labels)
                    ax.set_ylabel(ylabel)
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                plot_bar_line(axes_m[0], metric_data['ev'],   'Explained Variance (%)',        '% Variansi',           'lightblue',   'blue',       'o')
                plot_bar_line(axes_m[1], metric_data['mse'],  'Mean Squared Error (MSE)',       'Error (Lower is Better)', 'lightcoral', 'red',       's')

                axes_m[2].bar(x_pos, metric_data['psnr'], color='lightgreen', alpha=0.6)
                axes_m[2].plot(x_pos, metric_data['psnr'], color='green', marker='^',
                               linestyle='-', linewidth=2.5, markersize=8)
                axes_m[2].axhline(30, color='red', linestyle='--', linewidth=2, label='Standar Baik (30 dB)')
                axes_m[2].set_title('PSNR (Peak Signal-to-Noise Ratio)', fontweight='bold', pad=15)
                axes_m[2].set_xticks(x_pos); axes_m[2].set_xticklabels(k_labels)
                axes_m[2].set_ylabel('dB (Higher is Better)')
                axes_m[2].grid(True, alpha=0.3, axis='y')
                axes_m[2].spines['top'].set_visible(False)
                axes_m[2].spines['right'].set_visible(False)
                axes_m[2].legend()

                plot_bar_line(axes_m[3], metric_data['ssim'], 'SSIM',                           'SSIM (Mendekati 1 is Better)', 'thistle',    'purple',    'd')
                plot_bar_line(axes_m[4], metric_data['cr'],   'Compression Ratio',              'CR (Higher is Better)',        'navajowhite','darkorange','x')

                axes_m[5].axis('off')
                plt.tight_layout(pad=3.0)
                st.pyplot(fig_m)
                plt.close(fig_m)
            else:
                st.warning("📊 Masukkan **minimal 2 nilai K** untuk melihat grafik perbandingan metrik.")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses gambar: {str(e)}")
        st.exception(e)   # tampilkan traceback saat debug
else:
    st.info("👈 Silakan unggah gambar pada **Panel Pengaturan (Sidebar)** di sebelah kiri untuk memulai analisis PCA.")