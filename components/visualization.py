from typing import Dict, List, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from .component_feedback import create_feedback_container

class DataVisualizer:
    def create_feedback_container(self, title: str):
        """Create a container for feedback visualization with consistent styling."""
        return st.container()
        
    def plot_sentiment_distribution(self):
        """Plot the distribution of sentiment across different channels."""
        # Sample data - replace with actual data in production
        data = {
            'Channel': ['Twitter', 'Facebook', 'Reviews', 'Email'] * 3,
            'Sentiment': ['Positive', 'Neutral', 'Negative'] * 4,
            'Count': [20, 15, 5, 25, 10, 5, 30, 20, 10, 15, 25, 5]
        }
        df = pd.DataFrame(data)
        
        fig = px.bar(
            df,
            x='Channel',
            y='Count',
            color='Sentiment',
            barmode='group',
            title='Sentiment Distribution by Channel',
            color_discrete_map={
                'Positive': '#2ecc71',
                'Neutral': '#3498db',
                'Negative': '#e74c3c'
            }
        )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            xaxis_title="Channel",
            yaxis_title="Count"
        )
        
        with self.create_feedback_container("Sentiment Distribution") as container:
            st.plotly_chart(fig, use_container_width=True)
            
    def plot_performance_trends(self):
        """Plot performance trends over time."""
        # Sample data - replace with actual data in production
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
        data = {
            'Date': dates,
            'Accuracy': np.random.uniform(0.7, 0.95, len(dates)),
            'Response Time': np.random.uniform(0.8, 0.99, len(dates))
        }
        df = pd.DataFrame(data)
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Accuracy'],
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='#2ecc71', width=2)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Response Time'],
                mode='lines+markers',
                name='Response Time',
                line=dict(color='#3498db', width=2)
            )
        )
        
        fig.update_layout(
            title="Performance Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Score",
            height=400,
            showlegend=True,
            yaxis=dict(range=[0, 1])
        )
        
        with self.create_feedback_container("Performance Trends") as container:
            st.plotly_chart(fig, use_container_width=True)
            
    def plot_sentiment_details(self, sentiment_data: Dict):
        """Plot detailed sentiment analysis results."""
        if not sentiment_data:
            st.warning("No sentiment data available")
            return
            
        # Create gauge chart for sentiment score
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = sentiment_data.get('score', 0) * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2ecc71"},
                'steps': [
                    {'range': [0, 33], 'color': "#ffcdd2"},
                    {'range': [33, 66], 'color': "#fff9c4"},
                    {'range': [66, 100], 'color': "#c8e6c9"}
                ]
            },
            title = {'text': "Sentiment Score"}
        ))
        
        fig.update_layout(height=300)
        
        with self.create_feedback_container("Sentiment Score") as container:
            st.plotly_chart(fig, use_container_width=True)
            
        # Create bar chart for aspect-based sentiment
        if 'aspects' in sentiment_data:
            aspects_df = pd.DataFrame(sentiment_data['aspects'])
            fig = px.bar(
                aspects_df,
                x='aspect',
                y='score',
                title="Aspect-Based Sentiment",
                color='score',
                color_continuous_scale=['#e74c3c', '#f1c40f', '#2ecc71']
            )
            
            fig.update_layout(height=300)
            
            with self.create_feedback_container("Aspect Analysis") as container:
                st.plotly_chart(fig, use_container_width=True)
                
    def plot_brand_voice_analysis(self, analysis_data: Dict):
        """Plot brand voice analysis results using a radar chart."""
        categories = list(analysis_data.keys())
        values = list(analysis_data.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line=dict(color='#2ecc71', width=2),
            fillcolor='rgba(46, 204, 113, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title="Brand Voice Analysis"
        )
        
        with self.create_feedback_container("Brand Voice") as container:
            st.plotly_chart(fig, use_container_width=True)
            
    def plot_ml_performance(self, performance_data: Dict):
        """Plot ML model performance metrics."""
        if not performance_data or 'component_performance' not in performance_data:
            st.info("No ML performance data available")
            return
            
        components = performance_data['component_performance']
        
        # Prepare data for visualization
        data = {
            'Component': [],
            'Score': [],
            'Count': []
        }
        
        for component, metrics in components.items():
            data['Component'].append(component.replace('_', ' ').title())
            data['Score'].append(metrics['average_score'])
            data['Count'].append(metrics['count'])
            
        df = pd.DataFrame(data)
        
        # Create performance bar chart
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=df['Component'],
                y=df['Score'],
                marker_color='#2ecc71',
                text=df['Score'].apply(lambda x: f"{x:.1%}"),
                textposition='auto'
            )
        )
        
        fig.update_layout(
            title="ML Component Performance",
            xaxis_title="Component",
            yaxis_title="Average Score",
            yaxis=dict(range=[0, 1]),
            height=400,
            showlegend=False
        )
        
        with self.create_feedback_container("ML Performance") as container:
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data volume indicator
            volume_data = df['Count'].values
            if len(volume_data) > 0:
                avg_volume = np.mean(volume_data)
                volume_status = (
                    "High" if avg_volume >= 100 else
                    "Medium" if avg_volume >= 50 else
                    "Low"
                )
                st.metric(
                    "Data Volume Status",
                    volume_status,
                    help=f"Average {avg_volume:.0f} samples per component"
                )
                
    def plot_feedback_patterns(self, feedback_data: Dict):
        """Plot patterns in user feedback."""
        if not feedback_data or not feedback_data.get('feedback_patterns'):
            st.info("No feedback pattern data available")
            return
            
        patterns = feedback_data['feedback_patterns']
        
        # Create pattern visualization
        data = {
            'Pattern': list(patterns.keys()),
            'Frequency': list(patterns.values())
        }
        df = pd.DataFrame(data)
        
        fig = px.bar(
            df,
            x='Pattern',
            y='Frequency',
            title="Feedback Patterns",
            color='Frequency',
            color_continuous_scale=['#f1c40f', '#2ecc71']
        )
        
        fig.update_layout(
            xaxis_title="Pattern Type",
            yaxis_title="Frequency",
            height=400,
            showlegend=False
        )
        
        with self.create_feedback_container("Feedback Patterns") as container:
            st.plotly_chart(fig, use_container_width=True)
            
    def plot_performance_prediction(self, prediction_data: Dict):
        """Plot performance predictions with confidence intervals."""
        if not prediction_data or not prediction_data.get('predictions'):
            st.info("Insufficient data for performance prediction")
            return
            
        predictions = prediction_data['predictions']
        timestamps = prediction_data['timestamps']
        confidence = prediction_data['confidence']
        
        fig = go.Figure()
        
        # Add prediction line
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=predictions,
                mode='lines+markers',
                name='Predicted Performance',
                line=dict(color='#2ecc71', width=2),
                hovertemplate="Date: %{x}<br>" +
                            "Predicted Score: %{y:.2f}<br>" +
                            "<extra></extra>"
            )
        )
        
        # Add confidence interval
        confidence_range = 0.1 * (1 - confidence)  # Wider range for lower confidence
        upper_bound = [min(1.0, p + confidence_range) for p in predictions]
        lower_bound = [max(0.0, p - confidence_range) for p in predictions]
        
        fig.add_trace(
            go.Scatter(
                x=timestamps + timestamps[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(46, 204, 113, 0.2)',
                line=dict(color='rgba(46, 204, 113, 0)',),
                name='Confidence Interval',
                showlegend=True,
                hoverinfo='skip'
            )
        )
        
        fig.update_layout(
            title=f"Performance Prediction (Confidence: {confidence:.1%})",
            xaxis_title="Timeline",
            yaxis_title="Predicted Score",
            yaxis=dict(range=[0, 1]),
            hovermode='x unified',
            showlegend=True,
            height=400
        )
        
        with self.create_feedback_container("Performance Prediction") as container:
            st.plotly_chart(fig, use_container_width=True)
            
            # Display prediction insights
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Trend Direction",
                    prediction_data['trend_direction'].title(),
                    delta="Positive" if prediction_data['trend_direction'] == 'improving' else 
                          ("Negative" if prediction_data['trend_direction'] == 'declining' else "Stable")
                )
            with col2:
                st.metric(
                    "Trend Strength",
                    f"{prediction_data['trend_strength']:.2f}",
                    help="Magnitude of predicted change"
                )
            
            st.info(prediction_data['message'])

    def plot_sentiment_distribution(self, sentiments_data: Optional[Dict[str, float]] = None):
        """Plot sentiment distribution with enhanced visualization and insights."""
        if sentiments_data is None:
            # Generate sample sentiment data
            sentiments = ['positive', 'neutral', 'negative']
            values = [60, 30, 10]
        else:
            sentiments = list(sentiments_data.keys())
            values = list(sentiments_data.values())
            
        with self.create_feedback_container("Sentiment Distribution") as container:
        
            total = sum(values)
            percentages = [v/total*100 for v in values]
        
            # Create subplots for pie chart and bar chart
            fig = go.Figure()
        
            # Add pie chart
            fig.add_trace(
                go.Pie(
                    labels=sentiments,
                    values=values,
                    marker_colors=[self.color_scheme.get(s, '#6c757d') for s in sentiments],
                    textinfo='percent',
                    hovertemplate="<b>%{label}</b><br>" +
                                "Count: %{value}<br>" +
                                "Percentage: %{percent}<br>" +
                                "<extra></extra>",
                    hole=0.4,
                    domain={'x': [0, 0.5]}
                )
            )
        
            # Add bar chart
            fig.add_trace(
                go.Bar(
                    x=sentiments,
                    y=values,
                    marker_color=[self.color_scheme.get(s, '#6c757d') for s in sentiments],
                    hovertemplate="<b>%{x}</b><br>" +
                                "Count: %{y}<br>" +
                                "<extra></extra>"
                )
            )

            # Enhanced layout with side-by-side charts
            fig.update_layout(
                title={
                    'text': "Sentiment Analysis Overview",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                showlegend=False,
                grid={'rows': 1, 'columns': 2, 'pattern': 'independent'},
                annotations=[
                    {
                        'text': f'Total Analyzed<br>{total}',
                        'x': 0.25,
                        'y': 0.5,
                        'font_size': 16,
                        'showarrow': False
                    },
                    {
                        'text': f'Dominant Sentiment<br>{sentiments[percentages.index(max(percentages))].title()}',
                        'x': 0.25,
                        'y': 0.4,
                        'font_size': 14,
                        'showarrow': False
                    }
                ],
                margin=dict(l=20, r=20, t=60, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400
            )
        
            st.plotly_chart(fig, use_container_width=True)
        
            # Add insights below the chart
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Positive Ratio",
                    f"{percentages[sentiments.index('positive')]:.1f}%",
                    delta="good" if percentages[sentiments.index('positive')] > 50 else None
                )
            with col2:
                st.metric(
                    "Neutral Ratio",
                    f"{percentages[sentiments.index('neutral')]:.1f}%"
                )
            with col3:
                st.metric(
                    "Negative Ratio",
                    f"{percentages[sentiments.index('negative')]:.1f}%",
                    delta="down" if percentages[sentiments.index('negative')] > 30 else None
                )
        
    def plot_brand_voice_analysis(self, characteristics: Dict[str, float]):
        """Plot brand voice characteristics as an enhanced radar chart with insights."""
        categories = list(characteristics.keys())
        values = list(characteristics.values())
        
        # Calculate average and identify strengths/weaknesses
        avg_score = sum(values) / len(values)
        strengths = [cat for cat, val in characteristics.items() if val > avg_score + 0.1]
        weaknesses = [cat for cat, val in characteristics.items() if val < avg_score - 0.1]
        
        # Create enhanced radar chart
        fig = go.Figure()
        
        # Add the main radar plot with improved styling
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Current Profile',
            line=dict(color=self.color_scheme['positive'], width=2),
            fillcolor=f"rgba(40, 167, 69, 0.3)",  # Semi-transparent green
            hovertemplate="<b>%{theta}</b><br>" +
                         "Score: %{r:.2f}<br>" +
                         "<extra></extra>"
        ))
        
        # Add benchmark line (optional)
        fig.add_trace(go.Scatterpolar(
            r=[0.7] * len(categories),  # Benchmark level
            theta=categories,
            name='Benchmark',
            line=dict(color='rgba(128, 128, 128, 0.5)', dash='dash'),
            hovertemplate="Benchmark: 0.7<br>" +
                         "<extra></extra>"
        ))
        
        # Enhanced layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                    ticktext=['20%', '40%', '60%', '80%', '100%'],
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    linecolor='rgba(128, 128, 128, 0.2)'
                ),
                angularaxis=dict(
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    linecolor='rgba(128, 128, 128, 0.2)'
                ),
                bgcolor='rgba(255, 255, 255, 0.9)'
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=1.1,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.9)'
            ),
            title=dict(
                text="Brand Voice Analysis",
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=20)
            ),
            margin=dict(t=100, b=50),
            height=600
        )
        
        with self.create_feedback_container("Brand Voice Analysis") as container:
            # Display the enhanced radar chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insights section
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Overall Score",
                    f"{avg_score:.2f}",
                    delta="Above Target" if avg_score > 0.7 else ("At Target" if avg_score > 0.6 else "Below Target")
                )
            
            with col2:
                if strengths:
                    st.markdown("**Key Strengths**")
                    for strength in strengths[:3]:  # Top 3 strengths
                        st.markdown(f"• {strength}: {characteristics[strength]:.2f}")
            
            with col3:
                if weaknesses:
                    st.markdown("**Areas for Improvement**")
                    for weakness in weaknesses[:3]:  # Top 3 weaknesses
                        st.markdown(f"• {weakness}: {characteristics[weakness]:.2f}")
            
            # Add balance indicator
            variance = np.var(list(characteristics.values()))
            balance_status = (
                "🟢 Well Balanced" if variance < 0.02 
                else "🟡 Moderately Balanced" if variance < 0.05 
                else "🔴 Needs Balancing"
            )
            st.info(f"Voice Balance Status: {balance_status} (Variance: {variance:.3f})")

    def plot_sentiment_details(self, sentiment: Dict[str, float]):
        fig = go.Figure(data=[
            go.Bar(
                x=['Sentiment Score'],
                y=[sentiment['score']],
                marker_color=self.color_scheme[sentiment['label']]
            )
        ])
        
        fig.update_layout(
            title="Sentiment Analysis Score",
            yaxis_range=[0, 1]
        )
        
        with self.create_feedback_container("Sentiment Details") as container:
            st.plotly_chart(fig, use_container_width=True)

    def __init__(self):
        self.color_scheme: Dict[str, str] = {
            'positive': '#28a745',
            'neutral': '#ffc107',
            'negative': '#dc3545'
        }

    def plot_performance_trends(self):
        """Plot performance metrics with enhanced interactivity and tooltips."""
        # Generate sample data for demonstration
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        metrics = {
            'Engagement': np.random.normal(100, 15, 30).cumsum(),
            'Reach': np.random.normal(200, 25, 30).cumsum(),
            'Conversions': np.random.normal(50, 10, 30).cumsum(),
            'ROI': np.random.normal(75, 20, 30).cumsum()
        }
        
        df = pd.DataFrame(metrics, index=dates)
        
        # Calculate rolling averages and growth rates
        rolling_window = 7
        rolling_averages = df.rolling(window=rolling_window).mean()
        growth_rates = df.pct_change().rolling(window=rolling_window).mean() * 100
        
        fig = go.Figure()
        
        # Custom color scheme
        colors = {'Engagement': '#1f77b4', 'Reach': '#ff7f0e', 
                 'Conversions': '#2ca02c', 'ROI': '#d62728'}
        
        # Add main metric lines with enhanced tooltips and interactions
        for column in df.columns:
            growth = growth_rates[column].iloc[-1]
            growth_icon = "📈" if growth > 0 else "📉"
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[column],
                    name=column,
                    mode='lines+markers',
                    line=dict(width=2, color=colors[column]),
                    marker=dict(size=6),
                    hovertemplate=(
                        f"<b>{column}</b> {growth_icon}<br>" +
                        "Date: %{x|%Y-%m-%d}<br>" +
                        "Value: %{y:.2f}<br>" +
                        f"7-Day Growth: {growth:.1f}%<br>" +
                        "<extra></extra>"
                    )
                )
            )
            
            # Add rolling average trend lines with enhanced styling
            fig.add_trace(
                go.Scatter(
                    x=rolling_averages.index,
                    y=rolling_averages[column],
                    name=f"{column} Trend",
                    mode='lines',
                    line=dict(
                        dash='dot',
                        width=1,
                        color=colors[column]
                    ),
                    hoverinfo='skip',
                    showlegend=True,
                    opacity=0.5
                )
            )
        
        # Enhanced layout with better interactivity and annotations
        fig.update_layout(
            title={
                'text': "Performance Metrics Dashboard",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24)
            },
            xaxis_title="Timeline",
            yaxis_title="Metric Value",
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(211,211,211,0.5)",
                borderwidth=1
            ),
            margin=dict(l=20, r=20, t=80, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(211,211,211,0.3)',
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=14, label="2w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(211,211,211,0.3)',
                zeroline=True,
                zerolinecolor='rgba(211,211,211,0.5)',
                zerolinewidth=1
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": [True, True] * (len(df.columns))}],
                            label="Show All",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [True, False] * (len(df.columns))}],
                            label="Hide Trends",
                            method="restyle"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.98,
                    xanchor="right",
                    y=1.1,
                    yanchor="top"
                ),
            ]
        )
        
        # Add overall trend annotation
        latest_values = df.iloc[-1]
        max_metric = latest_values.idxmax()
        fig.add_annotation(
            text=f"Top Performer: {max_metric}",
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            font=dict(size=14, color="#2c3e50"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(211,211,211,0.5)",
            borderwidth=1,
            borderpad=4
        )
        
        # Render the chart with enhanced configuration and feedback
        with self.create_feedback_container("Performance Trends") as container:
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                    'toImageButtonOptions': {'height': None, 'width': None},
                }
            )

    def plot_ml_performance(self, performance_data: Dict):
        """Plot ML model performance metrics and trends."""
        if not performance_data or not performance_data.get('performance_trend'):
            st.info("Not enough feedback data to show performance metrics")
            return
        
        # Create performance trend chart
        trend_data = performance_data['performance_trend']
        dates = [pd.to_datetime(t['timestamp']) for t in trend_data]
        scores = [t['score'] for t in trend_data]
        
        fig = go.Figure()
        
        # Add performance trend line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=scores,
                mode='lines+markers',
                name='Performance Score',
                line=dict(color='#2ecc71', width=2),
                hovertemplate="Date: %{x}<br>" +
                            "Score: %{y:.2f}<br>" +
                            "<extra></extra>"
            )
        )
        
        # Add rolling average
        window = min(5, len(scores))
        if window > 1:
            rolling_avg = pd.Series(scores).rolling(window=window).mean()
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=rolling_avg,
                    mode='lines',
                    name=f'{window}-point Rolling Average',
                    line=dict(color='#3498db', width=2, dash='dot'),
                    hovertemplate="Date: %{x}<br>" +
                                "Average: %{y:.2f}<br>" +
                                "<extra></extra>"
                )
            )
        
        fig.update_layout(
            title="ML Model Performance Trend",
            xaxis_title="Timeline",
            yaxis_title="Performance Score",
            hovermode='x unified',
            showlegend=True,
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        container = self.create_feedback_container("ML Model Performance")
        with container:
            st.plotly_chart(fig, use_container_width=True)
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Average Score",
                f"{performance_data['average_feedback_score']:.2f}",
                delta="good" if performance_data['average_feedback_score'] > 0.7 else None
            )
        with col2:
            st.metric("Feedback Count", performance_data['feedback_count'])
        with col3:
            if len(scores) >= 2:
                recent_trend = scores[-1] - scores[-2]
                st.metric(
                    "Recent Trend",
                    f"{scores[-1]:.2f}",
                    delta=f"{recent_trend:+.2f}"
                )
