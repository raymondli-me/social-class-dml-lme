# UMAP v21 Application Guide

## Quick Start
1. Run: `python3 scripts/create_minimal_umap_viz_v21.py`
2. Open: `nvembed_dml_pc_analysis/minimal_umap_viz_v21.html`

## User Interface Guide

### Main View
- **3D Point Cloud**: Each point represents an essay
- **Auto-rotation**: Checkbox in right panel to toggle
- **Mouse Controls**: 
  - Left-click + drag: Rotate
  - Right-click + drag: Pan
  - Scroll: Zoom

### Color Modes (Center-left buttons)
1. **AI/SC**: Colors by AI rating and social class combinations
2. **Topics**: Colors by discovered topic clusters
3. **PC Gradient**: Purple-to-yellow gradient based on selected PC

### Panels (All support click-to-front)

#### Left Panel - Thresholds
- Adjust what counts as "high" or "low"
- Preset buttons for common percentiles
- Real-time count updates
- Collapsible with arrow button

#### Right Panel - Controls
- Auto-rotation speed
- Point cloud opacity
- Essay background opacity
- Show/hide various features
- Collapsible with arrow button

#### Bottom Panel - Essay Viewer
- Appears on hover
- Shows essay text with metadata
- Inline PC contributions (if enabled)
- Resizable from all edges/corners
- Layer toggle button
- Minimize button

#### Gallery Mode (Right buttons)
- Click category buttons to start
- Navigate with ← → arrows
- Auto-flies to each essay
- Shows current position (e.g., "3 of 47")
- ESC to exit

#### DML Statistics Table
- Check "Show DML Stats" to display
- Shows causal effect estimates
- Compares models with different controls
- Lists which PCs are used

#### Topic Statistics Panel (NEW)
- Check "Show Topic Stats" to display
- Shows % of essays in extremes for each topic
- Color coding: Green >30%, Yellow >15%, Gray <15%
- Ranked by maximum impact

#### PC Analysis Popup
- Click any PC name (e.g., in essay viewer)
- Shows detailed statistics
- Navigate between PCs with ← →
- ESC or × to close

### Keyboard Shortcuts
- **Gallery Mode**: ← → to navigate, ESC to exit
- **PC Analysis**: ← → to change PC, ESC to close
- **General**: Click panels to bring to front

### Understanding the Visualization

#### Categories (AI/SC Mode)
- **Green**: High AI rating + High social class
- **Magenta**: High AI rating + Low social class
- **Cyan**: Low AI rating + High social class  
- **Yellow**: Low AI rating + Low social class
- **Gray**: Middle values (not extreme)

#### Topics
- Text labels show discovered themes
- Slider controls how many are visible
- Based on essay content clustering

#### PC Gradient
- Shows how a specific PC varies across essays
- Purple = low percentile, Yellow = high percentile
- Legend shows variance explained

### Tips for Analysis

1. **Finding Patterns**: 
   - Look for spatial clustering of colors
   - Topics near colored regions show associations

2. **Understanding Extremes**:
   - Use gallery mode to read actual essays
   - Check topic stats to see which topics correlate with extremes

3. **Causal Insights**:
   - DML table shows how much social class affects AI rating
   - Compare naive vs controlled estimates

4. **PC Exploration**:
   - Click PC names to understand what they capture
   - High variance PCs are most informative

### Technical Details

- **Data**: 9,513 essays with AI ratings and social class
- **Embeddings**: NVEmbed 4096D → PCA 200D → UMAP 3D
- **Topics**: HDBSCAN clustering with c-TF-IDF keywords
- **Statistics**: XGBoost for contributions, DML for causal effects

### Troubleshooting

- **Performance**: Reduce visible topics if slow
- **Overlapping panels**: Click to bring desired panel to front
- **Can't see essays**: Check opacity isn't too low
- **Topics not showing**: Ensure "Show Topics" is checked