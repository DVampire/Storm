from typing import Any
import pandas_ta as ta
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Kline, Line, Grid
from copy import deepcopy

from pyecharts.globals import CurrentConfig
CurrentConfig.ONLINE_HOST = ""

def plot_kline(data: pd.DataFrame,
               title: str,
               timestamp_format: str,
               width=3.5,
               opacity=0.8,
               ) -> Any:

    data = deepcopy(data)

    assert "Date" in data.columns, "Date column not found in data"
    assert "Open" in data.columns, "Open column not found in data"
    assert "High" in data.columns, "High column not found in data"
    assert "Low" in data.columns, "Low column not found in data"
    assert "Close" in data.columns, "Close column not found in data"

    # Calculate the indicators
    data['sma_5'] = ta.sma(data["Close"], length=5)
    bbands = ta.bbands(data["Close"], length=5)
    data['bbl'] = bbands.iloc[:, 0]
    data['bbu'] = bbands.iloc[:, 2]
    date = pd.to_datetime(data["Date"]).apply(lambda x: x.strftime(timestamp_format)).tolist()
    values = data[["Open", "Close", "Low", "High"]].values.tolist()

    kline = (
        Kline()
        .add_xaxis(xaxis_data=date)
        .add_yaxis(
            series_name=title,
            y_axis=values,
            itemstyle_opts=opts.ItemStyleOpts(color="#00da3c", color0="#ec0000", border_color="#00da3c",
                                              border_color0="#ec0000"),
        )
        .set_global_opts(
            legend_opts=opts.LegendOpts(
                is_show=True
            ),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
            tooltip_opts=opts.TooltipOpts(is_show=False),
            visualmap_opts=opts.VisualMapOpts(
                is_show=False,
                dimension=2,
                series_index=4,
                is_piecewise=True,
                pieces=[
                    {"value": 1, "color": "#ec0000"},
                    {"value": -1, "color": "#00da3c"},
                ],
            ),
            axispointer_opts=opts.AxisPointerOpts(
                is_show=True,
                link=[{"xAxisIndex": "all"}],
                label=opts.LabelOpts(background_color="#777"),
            ),
            brush_opts=opts.BrushOpts(
                x_axis_index="all",
                brush_link="all",
                out_of_brush={"colorAlpha": 0.1},
                brush_type="lineX",
            ),
        )
    )

    line = (
        Line()
        .add_xaxis(xaxis_data=date)
        .add_yaxis(
            series_name="MA5",
            y_axis=data['sma_5'].values.tolist(),
            is_smooth=False,
            is_hover_animation=False,
            linestyle_opts=opts.LineStyleOpts(width=width, opacity=opacity),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_yaxis(
            series_name="BBL",
            y_axis=data['bbl'].values.tolist(),
            is_smooth=False,
            is_hover_animation=False,
            linestyle_opts=opts.LineStyleOpts(width=width, opacity=opacity),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_yaxis(
            series_name="BBU",
            y_axis=data['bbu'].values.tolist(),
            is_smooth=False,
            is_hover_animation=False,
            linestyle_opts=opts.LineStyleOpts(width=width, opacity=opacity),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(xaxis_opts=opts.AxisOpts(type_="category"))
    )

    overlap_kline_line = kline.overlap(line)

    # Grid Overlap + Bar
    grid_chart = Grid(
        init_opts=opts.InitOpts(
            width="600px",
            height="400px",
            animation_opts=opts.AnimationOpts(animation=False),
            bg_color="white",
        )
    )
    grid_chart.add(
        overlap_kline_line,
        grid_opts=opts.GridOpts(pos_left="10%", pos_right="10%", pos_top="20%", pos_bottom="10%", height="70%"),
    )

    return grid_chart