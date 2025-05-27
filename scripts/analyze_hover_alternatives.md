# Alternative Hover Solutions for UMAP Visualization

## The Core Problem
Three.js raycasting with Points + custom shaders + dynamic scaling = unreliable hover detection

## Alternative Approaches

### 1. **Switch to Instanced Mesh (Recommended)**
Instead of Points, use InstancedMesh with small spheres:
- Each point becomes a tiny sphere instance
- Much more reliable raycasting
- Better performance with filtering
- Exact hover detection

```javascript
// Instead of THREE.Points
const sphereGeometry = new THREE.SphereGeometry(0.5, 8, 6);
const instancedMesh = new THREE.InstancedMesh(sphereGeometry, material, data.length);

// Set position/scale for each instance
const matrix = new THREE.Matrix4();
data.forEach((d, i) => {
    matrix.makeTranslation(d.x * scale, d.y * scale, d.z * scale);
    matrix.scale(new THREE.Vector3(size, size, size));
    instancedMesh.setMatrixAt(i, matrix);
});
```

### 2. **Use Sprites Instead**
Replace points with sprites that always face camera:
- More control over appearance
- Better hover detection
- Can use images/textures

```javascript
data.forEach(d => {
    const spriteMaterial = new THREE.SpriteMaterial({ 
        color: getColor(d),
        sizeAttenuation: true 
    });
    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.position.set(d.x * scale, d.y * scale, d.z * scale);
    sprite.scale.set(size, size, 1);
    scene.add(sprite);
});
```

### 3. **2D Canvas Overlay Approach**
Project 3D points to 2D screen space and handle hover in 2D:
- Render points in 3D
- Track 2D positions
- Handle hover detection in 2D space

```javascript
// Project 3D to 2D
function updateScreenPositions() {
    data.forEach((d, i) => {
        const vector = new THREE.Vector3(d.x * scale, d.y * scale, d.z * scale);
        vector.project(camera);
        
        d.screenX = (vector.x + 1) / 2 * window.innerWidth;
        d.screenY = -(vector.y - 1) / 2 * window.innerHeight;
    });
}

// 2D hover detection
function onMouseMove(event) {
    const mouse = { x: event.clientX, y: event.clientY };
    
    let closest = null;
    let minDist = Infinity;
    
    data.forEach(d => {
        if (!d.visible) return;
        const dist = Math.sqrt(
            Math.pow(mouse.x - d.screenX, 2) + 
            Math.pow(mouse.y - d.screenY, 2)
        );
        if (dist < d.screenSize && dist < minDist) {
            minDist = dist;
            closest = d;
        }
    });
}
```

### 4. **Octree Spatial Index**
Build spatial index for faster/accurate detection:
- Divide space into octree
- Query only nearby points
- More efficient for large datasets

### 5. **GPU Picking**
Render scene to texture with unique colors per point:
- Each point gets unique color ID
- Read pixel under mouse
- Map color back to point

### 6. **Simplified Geometry Approach**
Use actual geometry instead of points:
- Small cubes or icosahedrons
- Better raycasting support
- More control over appearance

## Quick Fix Attempt

For the current implementation, try these adjustments:

```javascript
// 1. Increase raycaster line precision
raycaster.params.Points.threshold = 0.1;
raycaster.params.Line.threshold = 0.1;

// 2. Use different raycasting approach
function customRaycast(mouse, camera) {
    // Convert mouse to normalized device coordinates
    const mouseNDC = new THREE.Vector2(
        (mouse.x / window.innerWidth) * 2 - 1,
        -(mouse.y / window.innerHeight) * 2 + 1
    );
    
    // Create ray from camera
    const ray = new THREE.Ray();
    ray.origin.setFromMatrixPosition(camera.matrixWorld);
    ray.direction.set(mouseNDC.x, mouseNDC.y, 0.5)
        .unproject(camera)
        .sub(ray.origin)
        .normalize();
    
    // Manual distance checking
    let closest = null;
    let minDistance = Infinity;
    
    data.forEach((d, i) => {
        if (sizes[i] === 0) return;
        
        const point = new THREE.Vector3(
            positions[i * 3],
            positions[i * 3 + 1],
            positions[i * 3 + 2]
        );
        
        const distance = ray.distanceToPoint(point);
        const screenDistance = point.distanceTo(camera.position);
        const threshold = (sizes[i] * 50) / screenDistance; // Match shader
        
        if (distance < threshold && distance < minDistance) {
            minDistance = distance;
            closest = i;
        }
    });
    
    return closest;
}
```

## Recommendation

**Switch to InstancedMesh** - it's the most reliable solution:
1. Better hover detection
2. Works with all Three.js features
3. Good performance
4. No shader complexity for hit testing

Would you like me to implement one of these alternatives?