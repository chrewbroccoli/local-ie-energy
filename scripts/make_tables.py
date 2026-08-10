"""Emit Tables 2, 3 and 4 of the paper as LaTeX, from the shipped results.

    python scripts/make_tables.py

Writes tables/table2_parsing_energy.tex, tables/table3_vrdu.tex and
tables/table4_kleister.tex. These are the same numbers the figure scripts use --
both go through scripts/results.py -- so a table and a figure cannot disagree.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results as R


def fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def table2():
    pages = R.parser_page_counts()
    rows = []
    for tool, label in (("tesseract", "Tesseract"), ("docling", "Docling"),
                        ("deepseek", "DeepSeek-OCR 2")):
        cells = []
        for ds in ("Kleister-NDA", "VRDU"):
            rec = pages[(ds, tool)]
            cells += [f"{rec['total_wh']:.2f}", f"{rec['total_wh'] / rec['pages']:.4f}"]
        rows.append(f"        {label:<15}& " + " & ".join(cells) + r" \\")
    counts = {k: v["pages"] for k, v in pages.items()}
    return f"""\\begin{{table}}[htb]
    \\centering
    \\footnotesize
    \\caption{{Total and per-page energy consumption (Wh) for PDF to Markdown conversion, over 500 documents per dataset. Per-page figures use each tool's own page count, which differs because Tesseract rasterizes the PDFs ({counts[('Kleister-NDA','tesseract')]} pages on Kleister-NDA, {counts[('VRDU','tesseract')]} on VRDU) while Docling and DeepSeek-OCR~2 follow the PDF page structure ({counts[('Kleister-NDA','docling')]} and {counts[('VRDU','docling')]}).}}
    \\label{{tab:energy_consumption_integrated}}
    \\begin{{tabular}}{{@{{}} l c c c c @{{}}}}
        \\toprule
        & \\multicolumn{{2}}{{c}}{{\\textbf{{Kleister-NDA}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{VRDU}}}} \\\\
        \\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
        \\textbf{{Tool}} & \\textbf{{Total}} & \\textbf{{/Pg.}} & \\textbf{{Total}} & \\textbf{{/Pg.}} \\\\
        \\midrule
{chr(10).join(rows)}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}
"""


def detail_rows(ds, runs, arctic, nuextract, dagger):
    """Rows in the paper's order: models alphabetically, each with its parsers."""
    mark = "$^\\dagger$" if dagger else ""
    rows = []

    for parser in ("deepseek", "docling", "tesseract"):
        a = arctic.get((ds, parser))
        if not a:
            continue
        ocr = R.parser_energy_per_page(ds, parser)
        inf = a["inference_per_page"]
        e2e = "--" if inf is None else f"{ocr + inf:.1f}"
        rows.append(f"Arctic-TILT{mark} & 10 & {R.PARSER_LABEL[parser]} & "
                    f"{fmt(a['em'])} & {fmt(a['fuzzy'])} & {fmt(a['f1'])} & {ocr:.1f} & {e2e} \\\\")

    entries = []
    for m in R.TEXT_MODELS:
        for parser in ("deepseek", "docling", "tesseract"):
            r = R.select(runs, ds, m, parser)
            if not r:
                continue
            ocr = R.parser_energy_per_page(ds, parser)
            entries.append((m, f"{m} & 10 & {R.PARSER_LABEL[parser]} & {fmt(r['em'])} & "
                               f"{fmt(r['fuzzy'])} & {fmt(r['f1'])} & {ocr:.1f} & "
                               f"{ocr + r['energy_per_page']:.1f} \\\\"))
    n = nuextract.get(ds)
    if n:
        entries.append(("NuExtract-2.0-4B",
                        f"NuExtract-2.0-4B & 1 & - & {fmt(n['em'])} & {fmt(n['fuzzy'])} & "
                        f"{fmt(n['f1'])} & - & {n['energy_per_page']:.1f} \\\\"))
    for m in R.VISION_MODELS:
        r = R.select(runs, ds, m, "vision")
        if not r:
            continue
        entries.append((m, f"{m} & 10 & - & {fmt(r['em'])} & {fmt(r['fuzzy'])} & "
                           f"{fmt(r['f1'])} & - & {r['energy_per_page']:.1f} \\\\"))

    rows += [row for _, row in sorted(entries, key=lambda e: e[0])]
    return rows


def detail_table(ds, rows, label, caption, footnote=None):
    body = "\n".join(rows)
    foot = (f"\n\n\\vspace{{2pt}}\n{{\\footnotesize {footnote}}}" if footnote else "")
    return f"""\\begin{{table*}}[ht]
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{llcccccc}}
\\toprule
& & & & & & \\multicolumn{{2}}{{c}}{{Energy (mWh/pg)}} \\\\
\\cmidrule(lr){{7-8}}
Model & BS & OCR & EM (\\%) & FM (\\%) & F1 (\\%) & OCR & E2E \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}{foot}
\\end{{table*}}
"""


def main():
    runs = R.load_runs()
    R.report_duplicates()
    arctic, nuextract = R.load_arctic(), R.load_nuextract()
    R.TABLES.mkdir(exist_ok=True)

    (R.TABLES / "table2_parsing_energy.tex").write_text(table2())

    v = detail_table(
        "VRDU", detail_rows("VRDU", runs, arctic, nuextract, dagger=False),
        "tab:vrdu_plot_results",
        "Detailed benchmarking results for the VRDU dataset (Pareto plot configurations), "
        "over 500 documents. OCR~(mWh/pg) is the parsing energy per page and E2E~(mWh/pg) "
        f"additionally includes model inference; both are normalized by the "
        f"{R.pages_per_doc('VRDU'):.2f} pages per document of this dataset. "
        "The OCR text shipped with the VRDU benchmark "
        "is used for accuracy comparisons in Figure~\\ref{fig:input-modality-accuracy} but "
        "excluded here, since the energy of producing it is not attributable to our pipeline.")
    (R.TABLES / "table3_vrdu.tex").write_text(v)

    k = detail_table(
        "Kleister-NDA", detail_rows("Kleister-NDA", runs, arctic, nuextract, dagger=True),
        "tab:kleister_nda_plot_results",
        "Detailed benchmarking results for the Kleister-NDA dataset (Pareto plot "
        "configurations). Extraction is evaluated over 337 documents; parsing energy is "
        "amortized over the 500 documents of the parsing benchmark. Energy is per page, "
        f"normalized by the {R.pages_per_doc('Kleister-NDA'):.2f} pages per document of this "
        "dataset. Arctic-TILT is fine-tuned "
        "on Kleister-NDA by its authors and is therefore evaluated in-domain, unlike every "
        "other model in the table; its rows are reported for reference and are excluded from "
        "best-configuration and Pareto comparisons.",
        footnote="$^\\dagger$ In-domain: fine-tuned on Kleister-NDA by its authors; "
                 "not comparable to the zero-shot rows.")
    (R.TABLES / "table4_kleister.tex").write_text(k)

    for name in ("table2_parsing_energy", "table3_vrdu", "table4_kleister"):
        print(f"  wrote tables/{name}.tex")


if __name__ == "__main__":
    main()
