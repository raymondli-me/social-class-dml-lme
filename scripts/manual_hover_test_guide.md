# Manual Testing Guide for UMAP Hover Fix

## Test Setup
1. Open the fixed visualization: `umap_dml_top5_pcs_hover_fixed.html`
2. Enable debug mode by checking "Show Debug Info" checkbox
3. Keep the browser console open (F12)

## Test Cases

### 1. Basic Hover Alignment (CRITICAL)
**Steps:**
1. Hover over any visible point
2. Note the TID shown in tooltip
3. Check debug info shows matching world coordinates
4. Verify highlight sphere appears exactly on the point

**Expected:** Tooltip data matches the visual point being hovered

### 2. Scale Factor Test
**Steps:**
1. Set cloud scale to 0.5x
2. Hover over several points - verify alignment
3. Set cloud scale to 5.0x
4. Hover over same points - verify alignment
5. Set cloud scale to 10.0x
6. Hover over same points - verify alignment

**Expected:** Hover accuracy remains consistent at all scales

### 3. Hidden Point Test
**Steps:**
1. Uncheck "Lower (1)" and "Working (2)" social classes
2. Try hovering where those points were
3. Hover on visible points nearby
4. Re-enable all classes

**Expected:** Hidden points don't block hover on visible points behind them

### 4. PC Filter Test
**Steps:**
1. Select "PC 1 Extremes" from PC filter
2. Set percentile to 10%
3. Hover on visible extreme points
4. Try hovering in empty areas where middle points were

**Expected:** Only extreme points are hoverable, middle points truly hidden

### 5. Zoom and Rotation Test
**Steps:**
1. Zoom in close to a cluster of points
2. Test hover accuracy when zoomed in
3. Rotate the view 180 degrees
4. Test hover from different angles

**Expected:** Hover remains accurate regardless of camera position

### 6. Performance Test
**Steps:**
1. Move mouse rapidly across the visualization
2. Watch for tooltip lag or stuttering
3. Check browser console for errors

**Expected:** Smooth hover response, no console errors

### 7. Edge Cases
**Steps:**
1. Hover at the very edge of a point
2. Hover between closely spaced points
3. Hover with opacity set to 0.1
4. Hover with point size set to 20

**Expected:** Edge detection matches visual boundaries

## Debug Info Verification

When debug mode is enabled, verify:
- **Point Index:** Matches the hovered point
- **World Pos:** Coordinates that scale with cloud scale
- **Original Pos:** Base coordinates (unchanged)
- **Scale Factor:** Matches slider value
- **Point Size:** Matches size slider
- **Camera Dist:** Updates as you zoom

## Common Issues to Check

1. **Misaligned Hover**
   - Tooltip shows data for wrong point
   - Highlight sphere appears offset from cursor

2. **Dead Zones**
   - Areas where hover doesn't work
   - Hidden points blocking visible ones

3. **Scale Desync**
   - Hover accuracy degrades at certain scales
   - World coordinates don't update with scale

4. **Performance Issues**
   - Lag when moving mouse
   - Stuttering tooltip updates

## Test Results Template

```
Date: ___________
Tester: ___________
Browser: ___________

Basic Hover: [ ] Pass [ ] Fail - Notes: ___________
Scale Test: [ ] Pass [ ] Fail - Notes: ___________
Hidden Points: [ ] Pass [ ] Fail - Notes: ___________
PC Filter: [ ] Pass [ ] Fail - Notes: ___________
Zoom/Rotation: [ ] Pass [ ] Fail - Notes: ___________
Performance: [ ] Pass [ ] Fail - Notes: ___________
Edge Cases: [ ] Pass [ ] Fail - Notes: ___________

Overall Assessment: [ ] All tests pass [ ] Issues found

Issues Found:
1. ___________
2. ___________
3. ___________

Recommendations:
___________
```

## Automated Console Tests

Paste these commands in the browser console for quick checks:

```javascript
// Check if world coordinates exist
console.log('World coords present:', data[0].worldX !== undefined);

// Check current scale factor
console.log('Current scale:', currentScaleFactor);

// Test performRaycast function
console.log('Custom raycast:', typeof performRaycast === 'function');

// Count visible points
const visibleCount = geometry.attributes.size.array.filter(s => s > 0).length;
console.log('Visible points:', visibleCount, 'of', data.length);

// Check debug mode
console.log('Debug enabled:', document.getElementById('debug-mode').checked);
```