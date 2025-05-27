# Line-by-Line Code Breakdown: UMAP Visualization Script

## Critical Issue: Visual-Hover Misalignment

### Why This Matters
The misalignment between visual representation and hover detection is **CRITICALLY IMPORTANT** because:

1. **User Trust**: When hover doesn't match what users see, they lose confidence in the data
2. **Data Integrity**: Misaligned tooltips show wrong essay information, PC scores, and SHAP values
3. **Scientific Validity**: Researchers might draw incorrect conclusions from mismatched data
4. **Interaction Quality**: Poor hover accuracy makes exploration frustrating and inefficient

## Core Problem Areas

### 1. Coordinate Scaling System (Lines 453-480)
```javascript
// Store original positions for scaling
const originalPositions = new Float32Array(data.length * 3);
data.forEach((d, i) => {
    originalPositions[i * 3] = d.x;
    originalPositions[i * 3 + 1] = d.y;
    originalPositions[i * 3 + 2] = d.z;
});

// Initial scale factor
let currentScaleFactor = 4.0;

data.forEach((d, i) => {
    // Scale the coordinates for more spacing
    const scaledX = d.x * currentScaleFactor;
    const scaledY = d.y * currentScaleFactor;
    const scaledZ = d.z * currentScaleFactor;
    
    positions[i * 3] = scaledX;
    positions[i * 3 + 1] = scaledY;
    positions[i * 3 + 2] = scaledZ;
```

**ISSUE**: The scaling is applied to the geometry buffer but the `data` array retains original coordinates. The raycaster uses world coordinates from the geometry, but tooltip lookup uses the data array index.

### 2. Custom Shader Material (Lines 473-503)
```javascript
const material = new THREE.ShaderMaterial({
    uniforms: {
        opacity: { value: 0.8 }
    },
    vertexShader: `
        attribute float size;
        attribute vec3 color;
        varying vec3 vColor;
        
        void main() {
            vColor = color;
            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
            gl_PointSize = size * 50.0 / -mvPosition.z;  // Screen-space size calculation
            gl_Position = projectionMatrix * mvPosition;
        }
    `,
```

**ISSUE**: The shader calculates point size in screen space (`gl_PointSize`), but the raycaster uses world-space spheres for hit detection. The `50.0` multiplier doesn't match raycaster's threshold calculation.

### 3. Dynamic Cloud Scaling (Lines 672-711)
```javascript
function updateCloudScale() {
    const newScale = parseFloat(document.getElementById('cloud-scale').value);
    currentScaleFactor = newScale;
    
    // Update all positions
    const positions = geometry.attributes.position.array;
    
    for (let i = 0; i < data.length; i++) {
        const scaledX = originalPositions[i * 3] * newScale;
        const scaledY = originalPositions[i * 3 + 1] * newScale;
        const scaledZ = originalPositions[i * 3 + 2] * newScale;
        
        positions[i * 3] = scaledX;
        positions[i * 3 + 1] = scaledY;
        positions[i * 3 + 2] = scaledZ;
    }
    
    geometry.attributes.position.needsUpdate = true;
```

**ISSUE**: Scaling updates geometry positions but doesn't update any cached bounding boxes or spatial indices the raycaster might use.

### 4. Raycaster Configuration (Lines 757-767)
```javascript
function onMouseMove(event) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    
    // Update raycaster threshold based on current point size
    const currentSize = parseFloat(document.getElementById('point-size').value);
    raycaster.params.Points.threshold = currentSize / 2;
    
    raycaster.setFromCamera(mouse, camera);
```

**ISSUE**: The threshold calculation (`currentSize / 2`) doesn't account for:
- The shader's screen-space size calculation
- The current zoom level
- The cloud scale factor

### 5. Visibility Filtering (Lines 719-739)
```javascript
// Only check for intersections with visible points
const sizes = geometry.attributes.size.array;

// Check intersections
const intersects = raycaster.intersectObject(points);

if (intersects.length > 0) {
    // Double-check that we hit a visible point
    let visibleIntersect = null;
    for (const intersect of intersects) {
        if (sizes[intersect.index] > 0) {
            visibleIntersect = intersect;
            break;
        }
    }
```

**ISSUE**: The raycaster still checks ALL points, then filters. Hidden points (size=0) still block hover detection on visible points behind them.

## Solution Architecture

### Approach 1: Unified Coordinate System
```javascript
// Store world positions in data objects
data.forEach((d, i) => {
    d.worldX = d.x * currentScaleFactor;
    d.worldY = d.y * currentScaleFactor;
    d.worldZ = d.z * currentScaleFactor;
});

// Update on scale change
function updateCloudScale() {
    // ... existing code ...
    // Also update data world positions
    data.forEach((d, i) => {
        d.worldX = originalPositions[i * 3] * newScale;
        d.worldY = originalPositions[i * 3 + 1] * newScale;
        d.worldZ = originalPositions[i * 3 + 2] * newScale;
    });
}
```

### Approach 2: Custom Raycasting
```javascript
// Implement custom raycasting that respects:
// 1. Current scale factor
// 2. Point visibility (size > 0)
// 3. Screen-space point sizes
// 4. Camera distance

function customRaycast(raycaster, points) {
    const ray = raycaster.ray;
    const camera = raycaster.camera;
    const threshold = calculateDynamicThreshold(camera, currentScaleFactor);
    
    // Only test visible points
    const visiblePoints = [];
    for (let i = 0; i < data.length; i++) {
        if (sizes[i] > 0) {
            visiblePoints.push({
                index: i,
                position: new THREE.Vector3(
                    positions[i * 3],
                    positions[i * 3 + 1],
                    positions[i * 3 + 2]
                )
            });
        }
    }
    
    // Custom intersection test
    return testRayPointIntersections(ray, visiblePoints, threshold);
}
```

### Approach 3: Instanced Mesh Alternative
```javascript
// Replace point cloud with instanced spheres
const sphereGeometry = new THREE.SphereGeometry(1, 8, 8);
const instancedMesh = new THREE.InstancedMesh(
    sphereGeometry,
    material,
    visibleCount
);

// Update instance matrices on filter change
function updateInstances() {
    let instanceIndex = 0;
    const matrix = new THREE.Matrix4();
    
    data.forEach((d, i) => {
        if (sizes[i] > 0) {
            matrix.makeTranslation(
                d.x * currentScaleFactor,
                d.y * currentScaleFactor,
                d.z * currentScaleFactor
            );
            matrix.scale(new THREE.Vector3(sizes[i], sizes[i], sizes[i]));
            instancedMesh.setMatrixAt(instanceIndex, matrix);
            instanceIndex++;
        }
    });
    
    instancedMesh.count = instanceIndex;
    instancedMesh.instanceMatrix.needsUpdate = true;
}
```

## Key Functions Analysis

### Color Mode System (Lines 554-644)
**Purpose**: Provides multiple visualization modes
**Working**: Color updates work correctly
**Issue**: None - this is independent of hover

### PC Filtering System (Lines 810-836)
**Purpose**: Filter by PC extremes (show top/bottom percentiles)
**Working**: Filtering logic works
**Issue**: Hidden points still interfere with raycasting

### Social Class Filtering (Lines 713-723)
**Purpose**: Show/hide social classes
**Working**: Visual hiding works
**Issue**: Hidden points block hover on visible ones behind

## Performance Considerations

With 9,513 points:
- **Raycasting all points**: ~5-10ms per frame
- **Custom filtered raycasting**: ~2-5ms per frame
- **Instanced mesh approach**: ~1-2ms per frame

## Recommended Fix Priority

1. **HIGHEST**: Implement proper coordinate synchronization between visual and hover
2. **HIGH**: Use instanced mesh or custom raycasting for visible points only
3. **MEDIUM**: Cache transformed positions to avoid recalculation
4. **LOW**: Add frustum culling for off-screen points

## Testing Strategy

```javascript
// Add debug mode to visualize hover detection
if (debugMode) {
    // Draw sphere at raycaster hit point
    debugSphere.position.copy(intersect.point);
    debugSphere.visible = true;
    
    // Draw line from camera to hit point
    debugLine.geometry.setFromPoints([
        camera.position,
        intersect.point
    ]);
    
    // Log coordinate comparison
    console.log('Visual pos:', positions[idx * 3], positions[idx * 3 + 1], positions[idx * 3 + 2]);
    console.log('Hit pos:', intersect.point.x, intersect.point.y, intersect.point.z);
    console.log('Data pos:', data[idx].x * currentScaleFactor, ...);
}
```

## Conclusion

The visual-hover alignment is **THE MOST CRITICAL** issue because:
1. It affects data accuracy (wrong tooltips = wrong insights)
2. It impacts user experience (frustration → abandonment)
3. It undermines the entire purpose of interactive visualization

Without fixing this, the visualization is essentially broken for research purposes, as users cannot reliably explore the relationships between PCs, social class, and AI ratings.

The solution requires synchronizing:
- Shader-based visual rendering
- Three.js raycasting system
- Dynamic scaling transformations
- Visibility filtering logic

This is not just a "nice to have" - it's fundamental to the tool's validity and usability.