import io
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st

st.set_page_config(page_title='Bioassay Analysis App', layout='wide')

# 1. 全局字體設定 (完全保留原始設定)
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 工具函數：正則解析與數據清洗
# ==========================================
def extract_conc_and_unit(sample_name: str):
    if sample_name.lower() in ['basal', 'control']:
        return '–', ''
    match = re.search(r'([\d.]+)\s*([a-zA-Zμµ/]+)', sample_name)
    if match:
        return match.group(1), match.group(2)
    return sample_name, ''


def sort_groups_smart(g_list):
    """保證組別依 basal -> control -> 濃度由小到大 -> alone 排序"""

    def sort_key(g):
        g_lower = g.lower()
        if 'basal' in g_lower:
            return (0, 0.0)
        if 'control' in g_lower or 'ctrl' in g_lower:
            return (1, 0.0)
        if 'alone' in g_lower:
            return (3, 0.0)
        match = re.search(r'([\d.]+)', g)
        if match:
            return (2, float(match.group(1)))
        return (2, 9999.0)

    return sorted(g_list, key=sort_key)


def process_assay_sheet(file_source, sheet_name: str, calc_factor: float = 1.0):
    df_raw = pd.read_excel(file_source, sheet_name=sheet_name)
    df = df_raw.iloc[3:].copy()
    df.columns = [str(c).strip() for c in df_raw.iloc[2].values]

    df['Min'] = pd.to_numeric(df['Min'], errors='coerce')
    df['Max'] = pd.to_numeric(df['Max'], errors='coerce')
    df['Delta_OD'] = df['Max'] - df['Min']
    df['Final_Val'] = df['Delta_OD'] * calc_factor

    df_valid = df[df['Final_Val'].notna()].copy()
    df_valid['Raw_Sample'] = df_valid['Sample'].astype(str).str.strip()
    df_valid = df_valid[~df_valid['Raw_Sample'].str.contains(r'\(Avg', na=False)]
    df_valid['Clean_Sample'] = (
        df_valid['Raw_Sample']
        .str.replace(r'\s*\(Run\s*\d+\)', '', regex=True)
        .str.strip()
    )

    return df_valid


# ==========================================
# 介面側邊欄 (Sidebar)
# ==========================================
st.sidebar.title('⚙️ 實驗模式與檔案')

assay_mode = st.sidebar.radio(
    '選擇實驗類型 (Assay Mode)',
    ['Elastase Release', 'Superoxide Generation'],
)

if assay_mode == 'Elastase Release':
    calc_factor = 1.0
    default_y_label = r'Elastase release (OD$_{405}$)'
    default_stim_name = 'fMLF'
    default_y_max = 0.8
    default_y_step = 0.1
else:
    calc_factor = 47.4
    default_y_label = (
        'Superoxide generation\n' r'(nmol/6$\times 10^5$ cells/10 min)'
    )
    default_stim_name = 'fMLP'
    default_y_max = 0.0
    default_y_step = 0.0

uploaded_file = st.sidebar.file_uploader(
    '上傳 Excel 檔案 (.xlsx)', type=['xlsx', 'xls']
)

if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    skip_sheets = ['basal', 'NONE', '1']
    available_sheets = [s for s in xls.sheet_names if s not in skip_sheets]
    if not available_sheets:
        available_sheets = list(xls.sheet_names)

    selected_sheet = st.sidebar.selectbox('選擇藥物工作表 (Sheet)', available_sheets)
    df_clean = process_assay_sheet(uploaded_file, selected_sheet, calc_factor)

    # 智能依濃度升序排列組別
    raw_groups = list(df_clean['Clean_Sample'].unique())
    ordered_groups = sort_groups_smart(raw_groups)

    st.sidebar.markdown('---')
    st.sidebar.subheader('🧪 組別勾選')
    selected_groups = st.sidebar.multiselect(
        '選取欲呈現的組別',
        options=ordered_groups,
        default=ordered_groups,
    )

    if not selected_groups:
        st.warning('請至少選取一個組別進行繪圖！')
        st.stop()

    st.sidebar.markdown('---')
    st.sidebar.subheader('📐 統計檢定')
    stat_opt = st.sidebar.selectbox(
        '統計檢定方法',
        ["1: Dunnett's test vs Control", "2: Student's t-test", '3: 無 / 不標註'],
    )

    # 匯總計算
    summary_data = []
    raw_dict = {}
    detected_concs = []
    detected_units = []

    for g in selected_groups:
        vals = df_clean[df_clean['Clean_Sample'] == g]['Final_Val'].values
        raw_dict[g] = vals
        n = len(vals)
        mean_val = np.mean(vals) if n > 0 else 0
        sem_val = (np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0
        summary_data.append(
            {'Sample': g, 'mean': mean_val, 'sem': sem_val, 'count': n}
        )

        c_val, u_val = extract_conc_and_unit(g)
        detected_concs.append(c_val)
        if u_val:
            detected_units.append(u_val)

    sum_df = pd.DataFrame(summary_data)
    main_unit = (
        detected_units[0]
        if detected_units
        else ('μM' if assay_mode == 'Elastase Release' else 'μg/ml')
    )

    # 統計檢定
    p_values = {}
    ctrl_vals = raw_dict.get('control', np.array([]))
    for g in selected_groups:
        if g in ['basal', 'control']:
            continue
        vals = raw_dict.get(g, np.array([]))
        if len(ctrl_vals) >= 2 and len(vals) >= 2:
            if "Dunnett's" in stat_opt:
                res = stats.dunnett(
                    vals, control=ctrl_vals, rng=np.random.default_rng(42)
                )
                p_values[g] = res.pvalue[0]
            elif "Student's" in stat_opt:
                _, p_val = stats.ttest_ind(vals, ctrl_vals, equal_var=False)
                p_values[g] = p_val

    # ==========================================
    # 主面板：自訂設定區
    # ==========================================
    st.title(f'📊 {assay_mode} 統計圖產生器')

    with st.expander('🛠️ 彈性修改 X / Y 軸與矩陣橫排', expanded=True):
        col_y1, col_y2, col_y3 = st.columns([2, 1, 1])
        with col_y1:
            y_label_input = st.text_area(
                'Y 軸標籤文字 (支援 LaTeX 與 \\n)',
                value=default_y_label,
                height=70,
            )
            custom_drug_label = st.text_input(
                '矩陣第 1 排標籤 (藥物)', value=f'{selected_sheet} ({main_unit})'
            )
        with col_y2:
            y_max_custom = st.number_input(
                'Y 軸最高值 (0 為自動)',
                value=float(default_y_max),
                step=0.1 if assay_mode == 'Elastase Release' else 5.0,
            )
            stim_name = st.text_input('刺激劑名稱', value=default_stim_name)
        with col_y3:
            y_step_custom = st.number_input(
                'Y 軸刻度間距 (0 為自動)',
                value=float(default_y_step),
                step=0.05 if assay_mode == 'Elastase Release' else 1.0,
            )
            stim_conc = st.text_input(
                '刺激劑濃度', value=r'$10^{-7}$ M'
            )

        custom_stim_label = f'{stim_name} ({stim_conc})'

        # 預設前兩排
        st.markdown('**底部矩陣前兩排文字修改**')
        cols = st.columns(len(selected_groups))
        custom_concs = []
        custom_stims = []

        for i, col in enumerate(cols):
            with col:
                g = selected_groups[i]
                c_auto = detected_concs[i]
                val_c = col.text_input(
                    f'Bar {i+1} 濃度', value=c_auto, key=f'c_{i}'
                )
                s_auto = (
                    '–' if g in ['basal', f'{selected_sheet} (alone)'] else '+'
                )
                val_s = col.text_input(
                    f'Bar {i+1} 刺激劑', value=s_auto, key=f's_{i}'
                )
                custom_concs.append(val_c)
                custom_stims.append(val_s)

        # 動態增加自訂額外橫排
        st.markdown('---')
        st.markdown('**➕ 自訂額外矩陣橫排 (例如：Water, Vehicle, Inhibitor 等)**')

        if 'extra_rows_count' not in st.session_state:
            st.session_state.extra_rows_count = 0

        btn_c1, btn_c2, _ = st.columns([1, 1, 4])
        with btn_c1:
            if st.button('➕ 新增一排變數'):
                st.session_state.extra_rows_count += 1
                st.rerun()
        with btn_c2:
            if st.button('➖ 刪除最後一排') and st.session_state.extra_rows_count > 0:
                st.session_state.extra_rows_count -= 1
                st.rerun()

        extra_row_labels = []
        extra_row_data = []

        for r_num in range(st.session_state.extra_rows_count):
            r_c1, r_c2 = st.columns([1, 3])
            with r_c1:
                r_label = st.text_input(
                    f'額外排 {r_num+1} 標題',
                    value=f'Vehicle {r_num+1}',
                    key=f'extra_lbl_{r_num}',
                )
                extra_row_labels.append(r_label)
            with r_c2:
                r_cols = st.columns(len(selected_groups))
                row_vals = []
                for i, c in enumerate(r_cols):
                    with c:
                        v = st.text_input(
                            f'排 {r_num+1} - Bar {i+1}',
                            value='+',
                            key=f'extra_val_{r_num}_{i}',
                        )
                        row_vals.append(v)
                extra_row_data.append(row_vals)

    # ==========================================
    # 6. 開始繪製圖表 (100% 完全鎖定原始格式)
    # ==========================================
    total_matrix_rows = 3 + len(extra_row_labels)
    bottom_margin = min(0.48, 0.26 + (total_matrix_rows * 0.035))

    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    fig.subplots_adjust(left=0.25, bottom=bottom_margin)

    x_pos = np.arange(len(sum_df))
    line_width = 1.4

    bars = ax.bar(
        x_pos,
        sum_df['mean'],
        yerr=[np.zeros(len(sum_df)), sum_df['sem']],
        capsize=5,
        color='white',
        edgecolor='black',
        linewidth=line_width,
        width=0.5,
    )

    for line in ax.get_lines():
        line.set_linewidth(line_width)

    ax.set_ylabel(
        y_label_input,
        fontsize=18,
        labelpad=80 if assay_mode == 'Elastase Release' else 25,
    )

    # 計算 Y 軸上限與邊界
    max_bar_top = (sum_df['mean'] + sum_df['sem']).max()
    if y_max_custom > 0:
        y_top = float(y_max_custom)
    else:
        if max_bar_top <= 1.0:
            y_top = 0.8  # 固定上限為 0.8
        else:
            y_top = float(np.ceil(max_bar_top / 20) * 20)

    if y_step_custom > 0:
        y_step = float(y_step_custom)
    else:
        if y_top <= 1.0:
            y_step = 0.1
        elif y_top <= 40:
            y_step = 5.0
        elif y_top <= 100:
            y_step = 10.0
        else:
            y_step = 20.0

    y_bottom = -0.05 if sum_df['mean'].min() < 0 else 0.0
    ax.set_ylim(y_bottom, y_top)

    ax.set_yticks(np.arange(0, y_top + 0.0001, y_step))
    ax.set_xlim(-0.8, len(sum_df) - 0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)

    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_bounds(0, y_top)

    ax.tick_params(
        axis='y', direction='out', length=6, width=line_width, labelsize=18
    )
    ax.tick_params(axis='x', bottom=False)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])

    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_clip_on(False)

    # ==========================================
    # 7. 自動繪製動態矩陣表格
    # ==========================================
    all_row_labels = [custom_drug_label, custom_stim_label]
    all_col_data = [custom_concs, custom_stims]

    for lbl, data in zip(extra_row_labels, extra_row_data):
        all_row_labels.append(lbl)
        all_col_data.append(data)

    all_row_labels.append('Sample size')
    all_col_data.append([f'n={int(c)}' for c in sum_df['count']])

    y_offset = -0.08 * y_top
    y_step_offset = -0.09 * y_top

    for r_idx, label in enumerate(all_row_labels):
        ax.text(
            -0.9,
            y_offset + r_idx * y_step_offset,
            label,
            fontsize=18,
            ha='right',
            va='center',
        )

    for col_idx in range(len(x_pos)):
        for row_idx in range(len(all_row_labels)):
            ax.text(
                x_pos[col_idx],
                y_offset + r_idx * y_step_offset,
                all_col_data[row_idx][col_idx],
                fontsize=18,
                ha='center',
                va='center',
            )

    # ==========================================
    # 8. 標註星號 ⭐ (完全還原原始代碼的所有星號參數)
    # ==========================================
    for idx, row in sum_df.iterrows():
        sample = row['Sample']
        if sample in p_values:
            p_val = p_values[sample]
            star_str = (
                '***'
                if p_val < 0.001
                else '**'
                if p_val < 0.01
                else '*'
                if p_val < 0.05
                else ''
            )
            if star_str:
                y_anchor = row['mean'] + row['sem'] + (0.12 * y_top)
                x_adjusted = x_pos[idx] - 0.03
                ax.text(
                    x_adjusted,
                    y_anchor,
                    star_str,
                    fontsize=24,
                    ha='left',
                    va='center',
                    rotation=90,
                )

    # ==========================================
    # 9. 呈現與下載
    # ==========================================
    col1, col2 = st.columns([3, 2])
    with col1:
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300)
        st.download_button(
            label='💾 下載 300 DPI 高解析度圖檔 (PNG)',
            data=buf.getvalue(),
            file_name=f'{selected_sheet}_{assay_mode[:4]}_Plot.png',
            mime='image/png',
        )

    with col2:
        st.subheader('📋 數據摘要表')
        summary_display = sum_df.copy()
        summary_display['Mean ± SEM'] = summary_display.apply(
            lambda r: f"{r['mean']:.4f} ± {r['sem']:.4f}", axis=1
        )
        summary_display['p-value vs Control'] = summary_display['Sample'].map(
            lambda s: f'{p_values[s]:.4f}' if s in p_values else '–'
        )
        st.dataframe(
            summary_display[
                ['Sample', 'Mean ± SEM', 'count', 'p-value vs Control']
            ],
            use_container_width=True,
        )

else:
    st.info('👈 請由左側側邊欄上傳你的 Excel 數據檔。')
