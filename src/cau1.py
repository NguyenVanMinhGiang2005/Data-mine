import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# ====== Đọc dữ liệu ======
PATH = r"C:\Users\nguye\Documents\Khai thac du lieu\DOAN\D00 - wine.data.csv"

df = pd.read_csv(PATH)

# ====== Chuẩn bị Rich ======
console = Console()

# Khung thông tin nguồn
def thongTin():
    src_text = f"[bold]Tệp:[/bold] {PATH}"
    noneData = df.isnull().sum().sum()
    if noneData == 0:
        src_text += "\n[bold green]Dữ liệu đầy đủ, không thiếu ![/bold green]"
    else:
        src_text += f"\n[bold red]Dữ liệu thiếu:[/bold red] {noneData}"
    console.print(Panel.fit(src_text, title="Thông tin", box=box.ROUNDED))


# ======================================= A ====================================
# Khung kích thước
def cau1A():
    h, w = df.shape
    size_text = f"[bold]Số hàng:[/bold] {h}\n[bold]Số cột:[/bold] {w}"
    console.print(Panel.fit(size_text, title="Kích thước DataFrame", box=box.ROUNDED))


# ======================================= B ====================================
# Bảng kiểu dữ liệu
def cau1B():
    dtype_table = Table(box=box.SIMPLE_HEAVY)
    dtype_table.add_column("Cột", style="cyan", no_wrap=True)
    dtype_table.add_column("Kiểu dữ liệu", style="magenta")
    for col in df.columns:
        dtype_table.add_row(str(col), str(df[col].dtype))
    console.print(Panel.fit(dtype_table, title="Kiểu dữ liệu mỗi cột", box=box.ROUNDED))
    


# ======================================= C ====================================
def cau1C():
    label_col = None

    if label_col is None:
        label_col = df.columns[0]

    counts = df[label_col].value_counts(dropna=False).sort_index()
    unique_n = df[label_col].nunique(dropna=False)

    lines = [f"[bold]Cột nhãn:[/bold] {label_col}", f"[bold]Số nhãn khác nhau:[/bold] {unique_n}", ""]
    for idx, val in counts.items():
        label_str = "NaN" if pd.isna(idx) else str(idx)
        lines.append(f"[cyan]{label_str}[/cyan]: {val}")
    counts_text = "\n".join(lines)

    console.print(Panel.fit(counts_text, title="Số lượng thực thể theo nhãn", box=box.ROUNDED))

# ======================================= D ====================================

def cau1D():
    df_float = df.select_dtypes(include=['float', 'float32', 'float64'])
    stats = df_float.agg(['min', 'max', 'mean']).T.round(4)

    table = Table(title="Giá trị min / max / mean (chỉ cột số thực)")
    table.add_column("Cột", style="bold cyan")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Trung bình", justify="right")

    for col, row in stats.iterrows():
        table.add_row(col, str(row['min']), str(row['max']), str(row['mean']))

    console.print(table)

def main():
    thongTin()
    cau1A()
    cau1B()
    cau1C()
    cau1D()

if __name__ == "__main__":
    main()