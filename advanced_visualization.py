#!/usr/bin/env python3
"""
Advanced Visualization - Interactive and responsive visualization components.
Provides drill-down capabilities, custom views, dynamic filtering, and mobile optimization.
This module implements section 5.3 from the development plan.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Check Plotly version compatibility
REQUIRED_PLOTLY_VERSION = "5.18.0"
if plotly.__version__ != REQUIRED_PLOTLY_VERSION:
    print(f"⚠️ Warning: This module was designed for Plotly {REQUIRED_PLOTLY_VERSION}")
    print(f"Current version: {plotly.__version__}")
    print("Some visualizations may not display correctly due to API changes")
else:
    print(f"✅ Using Plotly {plotly.__version__}")

class AdvancedVisualization:
    """
    Advanced visualization components for the marketing ontology platform.
    Creates interactive and responsive visualizations with drill-down capabilities.
    """
    
    def __init__(self):
        """Initialize the AdvancedVisualization class."""
        # Ensure output directory exists
        Path("dashboard_data").mkdir(exist_ok=True)
        
        # Set color schemes
        self.color_scheme = {
            "primary": "#3498db",
            "secondary": "#2ecc71",
            "accent": "#e74c3c",
            "neutral": "#95a5a6",
            "dark": "#34495e",
            "channels": {
                "organic_search": "#3498db",
                "paid_search": "#2ecc71",
                "email": "#9b59b6",
                "social_media": "#e74c3c",
                "direct": "#f39c12",
                "referral": "#1abc9c"
            }
        }
        
        # Set default layout options
        self.layout_defaults = {
            "template": "plotly_white",
            "font": {"family": "Arial, sans-serif"},
            "margin": {"l": 40, "r": 40, "t": 50, "b": 40}
        }
    
    def load_dashboard_data(self, dashboard_type="executive"):
        """Load dashboard data from file."""
        try:
            file_path = f"dashboard_data/{dashboard_type}_dashboard_latest.json"
            if not os.path.exists(file_path):
                print(f"Dashboard data file not found: {file_path}")
                return None
                
            with open(file_path, 'r') as f:
                dashboard_data = json.load(f)
            
            return dashboard_data
        except Exception as e:
            print(f"Error loading dashboard data: {e}")
            return None
    
    def create_interactive_revenue_chart(self, growth_data, height=400, is_mobile=False):
        """Create an interactive revenue growth chart with drill-down capabilities."""
        if not growth_data or "monthly_data" not in growth_data:
            return self._create_blank_figure("No revenue growth data available")
        
        monthly_data = growth_data["monthly_data"]
        df = pd.DataFrame(monthly_data)
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add revenue bars
        fig.add_trace(
            go.Bar(
                x=df["month"],
                y=df["monthly_revenue"],
                name="Revenue",
                marker_color=self.color_scheme["primary"],
                customdata=df["monthly_purchases"],
                hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.2f}<br>Purchases: %{customdata:,}<extra></extra>"
            ),
            secondary_y=False
        )
        
        # Add customer line on secondary axis
        fig.add_trace(
            go.Scatter(
                x=df["month"],
                y=df["monthly_customers"],
                name="Customers",
                marker_color=self.color_scheme["accent"],
                mode="lines+markers",
                customdata=df["monthly_purchases"],
                hovertemplate="<b>%{x}</b><br>Customers: %{y:,}<br>Purchases: %{customdata:,}<extra></extra>",
                hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
            ),
            secondary_y=True
        )
        
        # Set layout
        title = "Revenue and Customer Growth"
        if is_mobile:
            # Simplified layout for mobile
            title = None  # Remove title on mobile
            
        fig.update_layout(
            title=title,
            height=height,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            **self.layout_defaults
        )
        
        # Update axes
        fig.update_xaxes(title_text="Month")
        fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
        fig.update_yaxes(title_text="Customers", secondary_y=True)
        
        # Add buttons for time period filtering
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.1,
                    y=1.15,
                    showactive=True,
                    buttons=[
                        dict(
                            label="YTD",
                            method="update",
                            args=[{"visible": [True, True]}, {"title": "Year to Date Growth"}]
                        ),
                        dict(
                            label="6M",
                            method="update",
                            args=[
                                {"visible": [True, True]},
                                {"title": "6 Month Growth"}
                            ]
                        ),
                        dict(
                            label="3M",
                            method="update",
                            args=[
                                {"visible": [True, True]},
                                {"title": "3 Month Growth"}
                            ]
                        ),
                        dict(
                            label="1M",
                            method="update",
                            args=[
                                {"visible": [True, True]},
                                {"title": "1 Month Growth"}
                            ]
                        ),
                        dict(
                            label="All",
                            method="update",
                            args=[
                                {"visible": [True, True]},
                                {"title": "Revenue and Customer Growth"}
                            ]
                        )
                    ]
                )
            ]
        )
        
        # Add annotations for growth rates
        if "revenue_cagr" in growth_data:
            revenue_cagr = growth_data["revenue_cagr"]
            customer_cagr = growth_data["customer_cagr"]
            
            # Only add annotations if not mobile
            if not is_mobile:
                fig.add_annotation(
                    x=0.02,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text=f"Revenue CAGR: {revenue_cagr:.1f}%",
                    showarrow=False,
                    font=dict(color=self.color_scheme["primary"]),
                    align="left",
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor=self.color_scheme["primary"],
                    borderwidth=1,
                    borderpad=4
                )
                
                fig.add_annotation(
                    x=0.02,
                    y=0.85,
                    xref="paper",
                    yref="paper",
                    text=f"Customer CAGR: {customer_cagr:.1f}%",
                    showarrow=False,
                    font=dict(color=self.color_scheme["accent"]),
                    align="left",
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor=self.color_scheme["accent"],
                    borderwidth=1,
                    borderpad=4
                )
        
        return fig
    
    def create_benchmark_radar_chart(self, benchmark_data, key_metrics=None, height=400, is_mobile=False):
        """Create an interactive radar chart for benchmarking."""
        if not benchmark_data or "metrics" not in benchmark_data:
            return self._create_blank_figure("No benchmark data available")
        
        metrics = benchmark_data["metrics"]
        
        # Use specified metrics or all metrics
        if key_metrics:
            metrics = [m for m in metrics if m["metric"] in key_metrics]
        
        # Prepare data for radar chart
        categories = [m["metric"] for m in metrics]
        company_values = [m["company"] for m in metrics]
        industry_values = [m["industry"] for m in metrics]
        best_values = [m["best"] for m in metrics]
        
        # For radar chart, normalize the values to make them comparable
        normalized_company = []
        normalized_industry = []
        normalized_best = []
        
        for i, metric in enumerate(metrics):
            metric_name = metric["metric"]
            company_val = metric["company"]
            industry_val = metric["industry"]
            best_val = metric["best"]
            
            # Determine if lower values are better for this metric
            lower_is_better = metric_name in ["Customer Acquisition Cost", "Cart Abandonment Rate"]
            
            # Get min and max
            if lower_is_better:
                min_val = min(company_val, industry_val, best_val)
                max_val = max(company_val, industry_val, best_val)
                # Invert normalization for metrics where lower is better
                normalized_company.append(1 - ((company_val - min_val) / (max_val - min_val)) if max_val > min_val else 0.5)
                normalized_industry.append(1 - ((industry_val - min_val) / (max_val - min_val)) if max_val > min_val else 0.5)
                normalized_best.append(1 - ((best_val - min_val) / (max_val - min_val)) if max_val > min_val else 0.5)
            else:
                min_val = min(company_val, industry_val, best_val)
                max_val = max(company_val, industry_val, best_val)
                normalized_company.append((company_val - min_val) / (max_val - min_val) if max_val > min_val else 0.5)
                normalized_industry.append((industry_val - min_val) / (max_val - min_val) if max_val > min_val else 0.5)
                normalized_best.append((best_val - min_val) / (max_val - min_val) if max_val > min_val else 0.5)
        
        # Create radar chart
        fig = go.Figure()
        
        # Add radar chart traces
        fig.add_trace(go.Scatterpolar(
            r=normalized_company,
            theta=categories,
            fill='toself',
            name='Your Company',
            line_color=self.color_scheme["primary"]
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_industry,
            theta=categories,
            fill='toself',
            name='Industry Average',
            line_color=self.color_scheme["neutral"]
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_best,
            theta=categories,
            fill='toself',
            name='Best in Class',
            line_color=self.color_scheme["secondary"]
        ))
        
        # Update layout
        title = "Industry Benchmarking"
        if is_mobile:
            title = None  # Remove title on mobile
        
        fig.update_layout(
            title=title,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            height=height,
            legend=dict(
                orientation="h" if not is_mobile else "v",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            **self.layout_defaults
        )
        
        # Add hover templates with actual values
        for i, trace in enumerate(fig.data):
            if i == 0:  # Company
                trace.customdata = company_values
                trace.hovertemplate = '%{theta}: %{customdata:.1f}<extra>Your Company</extra>'
            elif i == 1:  # Industry
                trace.customdata = industry_values
                trace.hovertemplate = '%{theta}: %{customdata:.1f}<extra>Industry Average</extra>'
            elif i == 2:  # Best
                trace.customdata = best_values
                trace.hovertemplate = '%{theta}: %{customdata:.1f}<extra>Best in Class</extra>'
        
        # Add dropdown to select metrics (desktop only)
        if not is_mobile and "historical" in metrics[0]:
            # Add dropdown to compare historical trends
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="dropdown",
                        direction="down",
                        x=0.1,
                        y=1.15,
                        showactive=True,
                        buttons=[
                            dict(
                                label="Current Benchmark",
                                method="update",
                                args=[{"visible": [True, True, True]}, {"title": "Industry Benchmarking"}]
                            ),
                            dict(
                                label="Historical Comparison",
                                method="update",
                                args=[
                                    {"visible": [True, False, False]},
                                    {"title": "Historical Performance Comparison"}
                                ]
                            )
                        ]
                    )
                ]
            )
        
        return fig
    
    def create_interactive_campaign_chart(self, campaign_data, height=400, is_mobile=False):
        """Create an interactive campaign performance chart with drill-down."""
        if not campaign_data or "campaigns" not in campaign_data:
            return self._create_blank_figure("No campaign data available")
        
        campaigns = campaign_data["campaigns"]
        df = pd.DataFrame(campaigns)
        
        if len(df) > 5:
            df = df.head(5)  # Limit to top 5 for readability
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add revenue bars
        fig.add_trace(
            go.Bar(
                x=df["campaign_name"],
                y=df["revenue"],
                name="Revenue",
                marker_color=self.color_scheme["primary"],
                text=df["revenue"].apply(lambda x: f"${x:,.2f}"),
                textposition="auto",
                customdata=df["campaign_id"],
                hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.2f}<br>ID: %{customdata}<extra></extra>",
                hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
            ),
            secondary_y=False
        )
        
        # Add conversion rate line on secondary axis
        fig.add_trace(
            go.Scatter(
                x=df["campaign_name"],
                y=df["conversion_rate"] * 100,  # Convert to percentage
                name="Conversion Rate",
                marker_color=self.color_scheme["accent"],
                mode="lines+markers",
                text=df["conversion_rate"].apply(lambda x: f"{x*100:.1f}%"),
                textposition="top center",
                customdata=df["campaign_id"],
                hovertemplate="<b>%{x}</b><br>CVR: %{y:.1f}%<br>ID: %{customdata}<extra></extra>",
                hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
            ),
            secondary_y=True
        )
        
        # Add click rate line on secondary axis
        fig.add_trace(
            go.Scatter(
                x=df["campaign_name"],
                y=df["click_rate"] * 100,  # Convert to percentage
                name="Click Rate",
                marker_color=self.color_scheme["secondary"],
                mode="lines+markers",
                text=df["click_rate"].apply(lambda x: f"{x*100:.1f}%"),
                textposition="top center",
                customdata=df["campaign_id"],
                hovertemplate="<b>%{x}</b><br>CTR: %{y:.1f}%<br>ID: %{customdata}<extra></extra>",
                hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
            ),
            secondary_y=True
        )
        
        # Set layout
        title = "Top Campaign Performance"
        if is_mobile:
            title = None  # Remove title on mobile
        
        fig.update_layout(
            title=title,
            height=height,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            **self.layout_defaults
        )
        
        # Update axes
        fig.update_xaxes(title_text="Campaign")
        fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
        fig.update_yaxes(title_text="Rate (%)", secondary_y=True)
        
        # Add buttons to toggle metrics
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.1,
                    y=1.15,
                    showactive=True,
                    buttons=[
                        dict(
                            label="Revenue",
                            method="update",
                            args=[{"visible": [True, False, False]}, {"title": "Campaign Revenue"}]
                        ),
                        dict(
                            label="Conversion Rate",
                            method="update",
                            args=[
                                {"visible": [False, True, False]},
                                {"title": "Campaign Conversion Rate"}
                            ]
                        ),
                        dict(
                            label="Click Rate",
                            method="update",
                            args=[
                                {"visible": [False, False, True]},
                                {"title": "Campaign Click Rate"}
                            ]
                        ),
                        dict(
                            label="All Metrics",
                            method="update",
                            args=[
                                {"visible": [True, True, True]},
                                {"title": "Campaign Performance"}
                            ]
                        )
                    ]
                )
            ]
        )
        
        return fig
    
    def create_channel_bubble_chart(self, channel_data, height=400, is_mobile=False):
        """Create an interactive bubble chart for channel performance."""
        if not channel_data or "channels" not in channel_data:
            return self._create_blank_figure("No channel data available")
        
        channels = channel_data["channels"]
        df = pd.DataFrame(channels)
        
        # Create bubble chart
        fig = go.Figure()
        
        # Default to darker colors for better contrast, but modify opacity for bubbles
        channel_colors = {}
        for channel_id in df["channel_id"]:
            if channel_id in self.color_scheme["channels"]:
                channel_colors[channel_id] = self.color_scheme["channels"][channel_id]
            else:
                # Generate a color for unknown channels
                channel_colors[channel_id] = self.color_scheme["neutral"]
        
        # Add bubble trace
        fig.add_trace(go.Scatter(
            x=df["conversion_rate"] * 100,  # Convert to percentage
            y=df["revenue_per_visitor"],
            size=df["visitors"],
            text=df["channel_id"],
            mode="markers",
            marker=dict(
                sizemode="area",
                sizeref=max(df["visitors"]) / 1000,
                color=[channel_colors.get(ch, self.color_scheme["neutral"]) for ch in df["channel_id"]],
                opacity=0.8,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            customdata=np.stack((
                df["channel_id"], 
                df["visitors"], 
                df["revenue"], 
                df["conversion_rate"] * 100, 
                df["revenue_per_visitor"],
                df["avg_order_value"]
            ), axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>" +
                         "Visitors: %{customdata[1]:,}<br>" +
                         "Revenue: $%{customdata[2]:,.2f}<br>" +
                         "Conversion: %{customdata[3]:.1f}%<br>" +
                         "Revenue/Visitor: $%{customdata[4]:.2f}<br>" +
                         "AOV: $%{customdata[5]:.2f}<extra></extra>",
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
        ))
        
        # Set layout
        title = "Channel Performance Comparison"
        if is_mobile:
            title = None  # Remove title on mobile
        
        fig.update_layout(
            title=title,
            xaxis_title="Conversion Rate (%)",
            yaxis_title="Revenue per Visitor ($)",
            height=height,
            template="plotly_white",
            **self.layout_defaults
        )
        
        # Add channel legend
        for channel_id in df["channel_id"]:
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(
                    size=10,
                    color=channel_colors.get(channel_id, self.color_scheme["neutral"])
                ),
                showlegend=True,
                name=channel_id
            ))
        
        # Interactive features - add segmentation options
        if not is_mobile:
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="right",
                        x=0.1,
                        y=1.15,
                        showactive=True,
                        buttons=[
                            dict(
                                label="All Channels",
                                method="update",
                                args=[{"visible": [True] + [True] * len(df["channel_id"])}, {"title": "Channel Performance Comparison"}]
                            ),
                            dict(
                                label="High Conversion",
                                method="update",
                                args=[
                                    {"visible": [True] + [True] * len(df["channel_id"])},
                                    {"title": "High Conversion Channels"}
                                ]
                            ),
                            dict(
                                label="High Value",
                                method="update",
                                args=[
                                    {"visible": [True] + [True] * len(df["channel_id"])},
                                    {"title": "High Value Channels"}
                                ]
                            )
                        ]
                    )
                ]
            )
        
        return fig
    
    def create_attribution_model_chart(self, channel_data, height=400, is_mobile=False):
        """Create an interactive attribution model comparison chart."""
        if not channel_data or "attribution_model" not in channel_data:
            return self._create_blank_figure("No attribution data available")
        
        attribution_data = channel_data["attribution_model"]
        
        # Extract model names and channel IDs
        models = [item["model"] for item in attribution_data]
        
        # Get all unique channels across all models
        all_channels = set()
        for model_data in attribution_data:
            for channel in model_data["channels"]:
                all_channels.add(channel["channel_id"])
        
        all_channels = sorted(list(all_channels))
        
        # Create a figure
        fig = go.Figure()
        
        # Add stacked bars for each model
        for model_idx, model_data in enumerate(attribution_data):
            # Create a dict to map channel to attribution
            model_name = model_data["model"]
            channel_attributions = {ch["channel_id"]: ch["attribution"] for ch in model_data["channels"]}
            
            # Ensure all channels are represented, fill in missing ones with zero
            for channel in all_channels:
                if channel not in channel_attributions:
                    channel_attributions[channel] = 0
            
            # Sort channels by attribution (descending) for more readable chart
            sorted_channels = sorted(
                channel_attributions.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Add bars for this model
            for i, (channel, attribution) in enumerate(sorted_channels):
                # Get channel color
                color = self.color_scheme["channels"].get(channel, self.color_scheme["neutral"])
                
                # Make bars narrower on mobile
                bar_width = 0.8 if not is_mobile else 0.6
                
                fig.add_trace(go.Bar(
                    x=[model_name],
                    y=[attribution],
                    name=channel,
                    marker_color=color,
                    text=f"{attribution:.1f}%",
                    textposition="inside",
                    hovertemplate=f"{channel}: %{{y:.1f}}%<extra></extra>",
                    showlegend=model_idx == 0,  # Only show in legend for the first model
                    width=bar_width
                ))
        
        # Set layout
        title = "Attribution Model Comparison"
        if is_mobile:
            title = None  # Remove title on mobile
        
        fig.update_layout(
            title=title,
            xaxis_title="Attribution Model",
            yaxis_title="Attribution (%)",
            barmode="stack",
            height=height,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.0,
                xanchor="right",
                x=1.1
            ),
            **self.layout_defaults
        )
        
        # Add buttons to toggle models (desktop only)
        if not is_mobile:
            buttons = []
            for model in models:
                buttons.append(
                    dict(
                        label=model,
                        method="update",
                        args=[
                            {"visible": [mod == model for mod in models for _ in range(len(all_channels))]},
                            {"title": f"{model} Attribution Model"}
                        ]
                    )
                )
            
            # Add "All Models" button
            buttons.append(
                dict(
                    label="All Models",
                    method="update",
                    args=[
                        {"visible": [True for _ in range(len(models) * len(all_channels))]},
                        {"title": "Attribution Model Comparison"}
                    ]
                )
            )
            
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="dropdown",
                        direction="down",
                        x=0.1,
                        y=1.15,
                        showactive=True,
                        buttons=buttons
                    )
                ]
            )
        
        return fig
    
    def create_service_metrics_dashboard(self, service_data, height=400, is_mobile=False):
        """Create interactive customer service metrics visualizations."""
        if not service_data or "summary" not in service_data:
            return self._create_blank_figure("No service metrics available")
        
        # Create figure with subplots
        if is_mobile:
            # For mobile, use a single column layout
            fig = make_subplots(
                rows=3, cols=1,
                specs=[
                    [{"type": "indicator"}],
                    [{"type": "pie"}],
                    [{"type": "scatter"}]
                ],
                subplot_titles=("Resolution Rate", "Ticket Categories", "Recent Trends"),
                vertical_spacing=0.2
            )
        else:
            # For desktop, use a more complex layout
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"type": "indicator"}, {"type": "pie"}],
                    [{"type": "bar"}, {"type": "scatter"}]
                ],
                subplot_titles=("Resolution Rate", "Ticket Categories", "Resolution Time by Priority", "Recent Trends"),
                vertical_spacing=0.15
            )
        
        # Extract data
        summary = service_data["summary"]
        categories = service_data["categories"]
        priority_distribution = service_data["priority_distribution"]
        resolution_time = service_data["resolution_time_by_priority"]
        trends = service_data["recent_trends"]
        
        # 1. Resolution Rate Gauge (row=1, col=1)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=summary["resolution_rate"],
                title={"text": "Resolution Rate"},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": self.color_scheme["primary"]},
                    "steps": [
                        {"range": [0, 60], "color": "#ffcccc"},
                        {"range": [60, 85], "color": "#ffffcc"},
                        {"range": [85, 100], "color": "#ccffcc"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 2},
                        "thickness": 0.75,
                        "value": 85
                    }
                },
                number={"suffix": "%", "font": {"size": 20}}
            ),
            row=1, col=1
        )
        
        # 2. Ticket Categories Pie Chart (row=1, col=2)
        category_labels = [cat["category"] for cat in categories]
        category_values = [cat["count"] for cat in categories]
        
        fig.add_trace(
            go.Pie(
                labels=category_labels,
                values=category_values,
                hole=0.4,
                textinfo="percent+label" if not is_mobile else "percent",
                insidetextorientation="radial",
                marker=dict(
                    colors=[
                        self.color_scheme["primary"],
                        self.color_scheme["secondary"],
                        self.color_scheme["accent"],
                        self.color_scheme["neutral"],
                        self.color_scheme["dark"]
                    ]
                )
            ),
            row=1, col=2
        )
        
        if not is_mobile:
            # 3. Resolution Time by Priority (row=2, col=1)
            fig.add_trace(
                go.Bar(
                    x=list(resolution_time.keys()),
                    y=list(resolution_time.values()),
                    marker_color=[
                        self.color_scheme["accent"],
                        self.color_scheme["primary"],
                        self.color_scheme["secondary"]
                    ],
                    text=[f"{hours:.1f}h" for hours in resolution_time.values()],
                    textposition="auto",
                    hovertemplate="Priority: %{x}<br>Resolution Time: %{y:.1f} hours<extra></extra>"
                ),
                row=2, col=1
            )
        
        # 4. Recent Trends (row=2, col=2 on desktop, row=3, col=1 on mobile)
        trend_days = [f"Day {i+1}" for i in range(len(trends["daily_tickets"]))]
        
        fig.add_trace(
            go.Scatter(
                x=trend_days,
                y=trends["daily_tickets"],
                name="New Tickets",
                mode="lines+markers",
                marker_color=self.color_scheme["accent"],
                hovertemplate="Day: %{x}<br>New Tickets: %{y}<extra></extra>"
            ),
            row=2 if not is_mobile else 3, col=2 if not is_mobile else 1
        )
        
        fig.add_trace(
            go.Scatter(
                x=trend_days,
                y=trends["daily_closed"],
                name="Closed Tickets",
                mode="lines+markers",
                marker_color=self.color_scheme["primary"],
                hovertemplate="Day: %{x}<br>Closed Tickets: %{y}<extra></extra>"
            ),
            row=2 if not is_mobile else 3, col=2 if not is_mobile else 1
        )
        
        # Set layout
        title = "Customer Service Metrics"
        if is_mobile:
            title = None  # Remove title on mobile
            height = 800  # Increase height for mobile
        
        fig.update_layout(
            title=title,
            height=height,
            **self.layout_defaults
        )
        
        # Add annotations with additional metrics
        if not is_mobile:
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.1, y=0.67,
                text=f"Open Tickets: {summary['open_tickets']}",
                showarrow=False,
                font=dict(size=12),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=self.color_scheme["accent"],
                borderwidth=1,
                borderpad=4
            )
            
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.1, y=0.6,
                text=f"Avg Resolution: {summary['avg_resolution_hours']:.1f}h",
                showarrow=False,
                font=dict(size=12),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=self.color_scheme["primary"],
                borderwidth=1,
                borderpad=4
            )
            
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.1, y=0.53,
                text=f"Satisfaction: {summary['avg_satisfaction']:.1f}/5",
                showarrow=False,
                font=dict(size=12),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=self.color_scheme["secondary"],
                borderwidth=1,
                borderpad=4
            )
        
        return fig
    
    def create_roi_analysis_dashboard(self, roi_data, height=400, is_mobile=False):
        """Create interactive ROI analysis visualizations."""
        if not roi_data:
            return self._create_blank_figure("No ROI data available")
        
        # Create figure with subplots
        if is_mobile:
            # For mobile, use a single column layout
            fig = make_subplots(
                rows=2, cols=1, 
                specs=[
                    [{"type": "bar"}],
                    [{"type": "scatter"}]
                ],
                subplot_titles=("Channel ROI", "ROI Trend"),
                vertical_spacing=0.2
            )
        else:
            # For desktop, use a more complex layout
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"colspan": 2, "type": "scatter"}, None]
                ],
                subplot_titles=("Channel ROI", "Campaign ROI", "ROI Trend"),
                vertical_spacing=0.15
            )
        
        # Extract data
        channel_roi = roi_data["channel_roi"]
        campaign_roi = roi_data["campaign_roi"]
        roi_trend = roi_data["roi_trend"]
        
        # 1. Channel ROI (row=1, col=1)
        channel_df = pd.DataFrame(channel_roi)
        
        fig.add_trace(
            go.Bar(
                x=channel_df["channel_id"],
                y=channel_df["roi"],
                marker_color=[self.color_scheme["channels"].get(ch, self.color_scheme["neutral"]) for ch in channel_df["channel_id"]],
                text=channel_df["roi"].apply(lambda x: f"{x:.1f}x"),
                textposition="auto",
                name="ROI",
                hovertemplate="Channel: %{x}<br>ROI: %{y:.2f}x<br>Cost: $%{customdata[0]:,.2f}<br>Revenue: $%{customdata[1]:,.2f}<extra></extra>",
                customdata=np.stack((channel_df["cost"], channel_df["revenue"]), axis=1)
            ),
            row=1, col=1
        )
        
        if not is_mobile:
            # 2. Campaign ROI (row=1, col=2)
            campaign_df = pd.DataFrame(campaign_roi)
            if len(campaign_df) > 5:
                campaign_df = campaign_df.head(5)  # Limit to top 5 for readability
            
            fig.add_trace(
                go.Bar(
                    x=campaign_df["campaign_name"],
                    y=campaign_df["roi"],
                    marker_color=self.color_scheme["primary"],
                    text=campaign_df["roi"].apply(lambda x: f"{x:.1f}x"),
                    textposition="auto",
                    name="Campaign ROI",
                    hovertemplate="Campaign: %{x}<br>ROI: %{y:.2f}x<br>Cost: $%{customdata[0]:,.2f}<br>Revenue: $%{customdata[1]:,.2f}<extra></extra>",
                    customdata=np.stack((campaign_df["cost"], campaign_df["revenue"]), axis=1)
                ),
                row=1, col=2
            )
        
        # 3. ROI Trend (row=2, col=1+2 on desktop, row=2, col=1 on mobile)
        trend_df = pd.DataFrame(roi_trend)
        
        fig.add_trace(
            go.Scatter(
                x=trend_df["month"],
                y=trend_df["roi"],
                mode="lines+markers",
                marker_color=self.color_scheme["primary"],
                name="ROI",
                hovertemplate="Month: %{x}<br>ROI: %{y:.2f}x<br>Cost: $%{customdata[0]:,.2f}<br>Revenue: $%{customdata[1]:,.2f}<extra></extra>",
                hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
                customdata=np.stack((trend_df["cost"], trend_df["revenue"]), axis=1)
            ),
            row=2, col=1 if is_mobile else 1
        )
        
        # Set layout
        title = "Marketing ROI Analysis"
        if is_mobile:
            title = None  # Remove title on mobile
        
        fig.update_layout(
            title=title,
            height=height,
            **self.layout_defaults
        )
        
        # Add annotations with overall metrics
        if not is_mobile:
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                text=f"Overall ROI: {roi_data['overall_roi']:.2f}x",
                showarrow=False,
                font=dict(size=14, color=self.color_scheme["primary"]),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=self.color_scheme["primary"],
                borderwidth=1,
                borderpad=4
            )
            
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.93,
                text=f"Total Cost: ${roi_data['total_cost']:,.2f}",
                showarrow=False,
                font=dict(size=12),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                borderpad=4
            )
            
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.88,
                text=f"Total Revenue: ${roi_data['total_revenue']:,.2f}",
                showarrow=False,
                font=dict(size=12),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                borderpad=4
            )
        
        # Label axes
        fig.update_yaxes(title_text="ROI (x)" if not is_mobile else None, row=1, col=1)
        
        if not is_mobile:
            fig.update_yaxes(title_text="ROI (x)", row=1, col=2)
        
        fig.update_yaxes(title_text="ROI (x)" if not is_mobile else None, row=2, col=1 if is_mobile else 1)
        
        return fig
    
    def create_drilldown_customer_segments(self, customer_data, height=400, is_mobile=False):
        """Create interactive customer segment visualizations with drill-down."""
        if not customer_data or "segments" not in customer_data:
            return self._create_blank_figure("No customer segment data available")
        
        segments = customer_data["segments"]
        
        # Create a figure with subplots
        if is_mobile:
            # Single column layout for mobile
            fig = make_subplots(
                rows=2, cols=1,
                specs=[
                    [{"type": "pie"}],
                    [{"type": "bar"}]
                ],
                subplot_titles=("Customer Segments", "Segment Metrics"),
                vertical_spacing=0.2
            )
        else:
            # More complex layout for desktop
            fig = make_subplots(
                rows=1, cols=2,
                specs=[
                    [{"type": "pie"}, {"type": "bar"}]
                ],
                subplot_titles=("Customer Segments", "Segment Metrics"),
                horizontal_spacing=0.1
            )
        
        # 1. Segment Distribution Pie Chart
        segment_labels = [seg["name"] for seg in segments]
        segment_values = [seg["count"] for seg in segments]
        
        fig.add_trace(
            go.Pie(
                labels=segment_labels,
                values=segment_values,
                hole=0.4,
                textinfo="percent+label" if not is_mobile else "percent",
                insidetextorientation="radial",
                marker=dict(
                    colors=[
                        self.color_scheme["primary"],
                        self.color_scheme["secondary"],
                        self.color_scheme["accent"],
                        self.color_scheme["neutral"],
                        self.color_scheme["dark"]
                    ]
                ),
                customdata=[seg["percentage"] for seg in segments],
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{customdata:.1f}%<extra></extra>"
            ),
            row=1, col=1
        )
        
        # 2. Segment Bar Chart (with drill-down capability)
        # In a real implementation, we would have additional metrics per segment
        # For this example, we'll create sample metrics
        sample_metrics = {
            "Customer Lifetime Value": [120, 75, 40, 20],
            "Purchase Frequency": [12, 8, 3, 1],
            "Average Order Value": [90, 60, 40, 30],
            "Churn Rate": [5, 10, 25, 40]
        }
        
        # Default to CLV for initial bar chart
        metric = "Customer Lifetime Value"
        metric_values = sample_metrics[metric]
        
        # Add Bar Chart
        fig.add_trace(
            go.Bar(
                x=segment_labels,
                y=metric_values[:len(segment_labels)],  # Ensure we don't use more values than segments
                marker_color=self.color_scheme["primary"],
                name=metric,
                text=[f"{val:.1f}" for val in metric_values[:len(segment_labels)]],
                textposition="auto",
                hovertemplate="<b>%{x}</b><br>%{name}: %{y}<extra></extra>"
            ),
            row=1 if not is_mobile else 2, col=2 if not is_mobile else 1
        )
        
        # Set layout
        title = "Customer Segment Analysis"
        if is_mobile:
            title = None  # Remove title on mobile
        
        fig.update_layout(
            title=title,
            height=height,
            **self.layout_defaults
        )
        
        # Add dropdown menu for metric selection (desktop only)
        if not is_mobile:
            buttons = []
            for i, (metric_name, metric_vals) in enumerate(sample_metrics.items()):
                buttons.append(
                    dict(
                        label=metric_name,
                        method="update",
                        args=[
                            {
                                "y": [None, metric_vals[:len(segment_labels)]],
                                "text": [None, [f"{val:.1f}" for val in metric_vals[:len(segment_labels)]]],
                                "name": [None, metric_name]
                            },
                            {"title": f"Customer Segment Analysis - {metric_name}"}
                        ]
                    )
                )
            
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="dropdown",
                        direction="down",
                        x=0.55,
                        y=1.15,
                        showactive=True,
                        buttons=buttons
                    )
                ]
            )
            
            # Add annotation for the dropdown
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.55, y=1.2,
                text="Select Metric:",
                showarrow=False,
                font=dict(size=12),
                align="center"
            )
        
        return fig
    
    def _create_blank_figure(self, message="No data available"):
        """Create a blank figure with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#95a5a6")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig
    
    def generate_dashboard_visualizations(self, is_mobile=False):
        """
        Generate all dashboard visualizations and save them to files.
        This creates JSON-encoded plotly figures for web embedding.
        """
        try:
            visualizations = {}
            
            # Load dashboard data
            executive_data = self.load_dashboard_data("executive")
            operational_data = self.load_dashboard_data("operational")
            
            if not executive_data or not operational_data:
                print("Dashboard data not available. Run executive_dashboard.py and operational_dashboard.py first.")
                return False
            
            # Executive Dashboard Visualizations
            visualizations["executive"] = {
                "revenue_growth": self.create_interactive_revenue_chart(
                    executive_data["strategic_kpis"]["growth"],
                    height=400 if not is_mobile else 350,
                    is_mobile=is_mobile
                ),
                "benchmark_radar": self.create_benchmark_radar_chart(
                    executive_data["benchmarking"],
                    height=400 if not is_mobile else 350,
                    is_mobile=is_mobile
                ),
                "customer_segments": self.create_drilldown_customer_segments(
                    executive_data["strategic_kpis"]["customers"],
                    height=400 if not is_mobile else 350,
                    is_mobile=is_mobile
                )
            }
            
            # Operational Dashboard Visualizations
            visualizations["operational"] = {
                "campaign_performance": self.create_interactive_campaign_chart(
                    operational_data["marketing_performance"],
                    height=400 if not is_mobile else 350,
                    is_mobile=is_mobile
                ),
                "channel_performance": self.create_channel_bubble_chart(
                    operational_data["channel_analysis"],
                    height=400 if not is_mobile else 350,
                    is_mobile=is_mobile
                ),
                "attribution_model": self.create_attribution_model_chart(
                    operational_data["channel_analysis"],
                    height=400 if not is_mobile else 350,
                    is_mobile=is_mobile
                ),
                "service_metrics": self.create_service_metrics_dashboard(
                    operational_data["service_metrics"],
                    height=500 if not is_mobile else 800,
                    is_mobile=is_mobile
                ),
                "roi_analysis": self.create_roi_analysis_dashboard(
                    operational_data["roi_analysis"],
                    height=500 if not is_mobile else 600,
                    is_mobile=is_mobile
                )
            }
            
            # Save visualizations to file
            self._save_visualizations(visualizations, is_mobile)
            
            return visualizations
            
        except Exception as e:
            print(f"Error generating dashboard visualizations: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_visualizations(self, visualizations, is_mobile=False):
        """Save dashboard visualizations to file."""
        try:
            device_type = "mobile" if is_mobile else "desktop"
            file_path = f"dashboard_data/visualizations_{device_type}.json"
            
            # Convert plotly figures to JSON
            json_visualizations = {}
            
            for dashboard_type, dashboard_visuals in visualizations.items():
                json_visualizations[dashboard_type] = {}
                
                for visual_name, visual_figure in dashboard_visuals.items():
                    # Convert to JSON and decode back to dict for compact storage
                    json_visualizations[dashboard_type][visual_name] = json.loads(visual_figure.to_json())
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(json_visualizations, f)
                
            print(f"Saved {device_type} visualizations to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error saving visualizations: {e}")
            return False

def main():
    """Generate dashboard visualizations when run as a script."""
    print("Generating dashboard visualizations...")
    visualization = AdvancedVisualization()
    
    # Generate desktop visualizations
    print("Creating desktop visualizations...")
    desktop_visuals = visualization.generate_dashboard_visualizations(is_mobile=False)
    
    # Generate mobile visualizations
    print("Creating mobile visualizations...")
    mobile_visuals = visualization.generate_dashboard_visualizations(is_mobile=True)
    
    if desktop_visuals and mobile_visuals:
        print("Successfully generated all dashboard visualizations")
    else:
        print("Failed to generate some dashboard visualizations")

if __name__ == "__main__":
    main()