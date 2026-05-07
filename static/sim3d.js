/**
 * sim3d.js — three.js front-end for the 3D climbing simulator.
 *
 * Talks to the Flask blueprint at /sim3d/api/*. Holds, the wall, and
 * the climber are all drawn from pose snapshots delivered by the
 * server. We don't run any physics in the browser — three.js only
 * reflects what MuJoCo computed.
 *
 * Coordinate system mirrors sim3d/__init__.py:
 *     +X = along the wall, climber's right
 *     +Y = away from the wall, toward the camera
 *     +Z = up
 * three.js's default has +Y as up, so we set scene.up to (0,0,1) and
 * point the perspective camera accordingly.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const LIMBS = ['LH', 'RH', 'LF', 'RF'];
// Mapping from MJCF body name → renderable (mesh) we keep in the scene.
// We rebuild this whenever a session is created/recreated.
let bodyMeshes = {};
let holdMeshes = {};
let holdRings = {};       // start/finish markers
let session = null;       // {id, wall, profile, holdsByID}
let pollTimer = null;
let liveTimer = null;

// ─── three.js scene ────────────────────────────────────────────────────────
const canvas = document.getElementById('canvas');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
scene.fog = new THREE.Fog(0x0d1117, 6, 18);
scene.up = new THREE.Vector3(0, 0, 1);

const camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.05, 100);
camera.up.set(0, 0, 1);
camera.position.set(0, 4.5, 1.5);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0, 1.4);
controls.enableDamping = true;

// Ambient + directional lighting. Climbing gyms tend to be brightly
// lit; we use a fill light at the camera to keep faces visible from
// any orbit position.
scene.add(new THREE.AmbientLight(0xffffff, 0.45));
const sun = new THREE.DirectionalLight(0xffffff, 0.8);
sun.position.set(2, 3, 5);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
sun.shadow.camera.left = -4;
sun.shadow.camera.right = 4;
sun.shadow.camera.top = 4;
sun.shadow.camera.bottom = -4;
scene.add(sun);

// Floor — large dark plane just below the wall.
const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(40, 40),
    new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.9 })
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.05;
floor.receiveShadow = true;
// Convert from three.js's default Y-up flat plane to our Z-up world:
// The plane is in XY by default after rotating; we want it on world XY.
// Easier: just place it in world XY directly.
floor.rotation.set(0, 0, 0);
floor.position.set(0, 0, -0.05);
scene.add(floor);

// Grid helper for orientation while debugging.
const grid = new THREE.GridHelper(8, 16, 0x444444, 0x202020);
grid.rotation.x = Math.PI / 2;
grid.position.set(0, 0, 0);
scene.add(grid);

// ─── Resize ────────────────────────────────────────────────────────────────
function resize() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

// ─── Render loop ───────────────────────────────────────────────────────────
function tick() {
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
}
tick();

// ─── Climber rendering ────────────────────────────────────────────────────
// A single MuJoCo body becomes one Object3D in three.js. Geometry is
// chosen to roughly match what the MJCF builder draws: capsules for
// limb segments, boxes for torso + hands + feet, sphere for head.
function buildClimberMeshes(profile) {
    // Approximate segment dimensions from the climber profile.
    // Should mirror sim3d/builder.py — these are visual stand-ins, not
    // physics. Off-by-a-few-cm is OK; the limb pose data drives the
    // actual position.
    const skin = 0xd9b88c;
    const cloth = 0x3366b3;
    const hMat = (col) => new THREE.MeshStandardMaterial({ color: col, roughness: 0.6 });
    const h = profile.height_cm / 100;
    const armTotal = Math.max(0.10, (profile.wingspan_cm / 100 - h * 0.23) / 2);
    const upperArm = armTotal * 0.55;
    const forearm = armTotal * 0.45;
    const thigh = h * 0.245;
    const shin = h * 0.245;
    const spine = h * 0.30;
    const head = h * 0.18;
    const sw = h * 0.23;          // shoulder width
    const pw = h * 0.18;          // pelvis width

    // Helper: build a capsule along local -Z (matches MJCF capsule fromto).
    const cap = (length, radius, mat) => {
        const g = new THREE.CapsuleGeometry(radius, Math.max(length - 2 * radius, 0.001), 4, 8);
        const m = new THREE.Mesh(g, mat);
        m.geometry.translate(0, -length / 2, 0);   // anchor at top
        m.rotation.x = Math.PI / 2;                // align with -Z
        m.castShadow = true;
        return m;
    };

    const meshes = {};
    // Pelvis — anchored to world coords; everything else gets transformed
    // each frame via xpos/xquat from the server.
    const pelvis = new THREE.Group();
    pelvis.add(new THREE.Mesh(
        new THREE.BoxGeometry(pw * 2, 0.16, 0.16),
        hMat(cloth),
    ));
    meshes.pelvis = pelvis;

    const chest = new THREE.Group();
    const chestBox = new THREE.Mesh(
        new THREE.BoxGeometry(sw * 1.8, 0.20, spine),
        hMat(cloth),
    );
    chestBox.position.z = spine / 2;
    chest.add(chestBox);
    meshes.chest = chest;

    const headG = new THREE.Group();
    headG.add(new THREE.Mesh(new THREE.SphereGeometry(head / 2, 16, 12), hMat(skin)));
    meshes.head = headG;

    for (const side of ['l', 'r']) {
        const upG = new THREE.Group();
        upG.add(cap(upperArm, 0.045, hMat(skin)));
        meshes[`${side}_upperarm`] = upG;

        const foreG = new THREE.Group();
        foreG.add(cap(forearm, 0.038, hMat(skin)));
        meshes[`${side}_forearm`] = foreG;

        const handG = new THREE.Group();
        handG.add(new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.05, 0.12), hMat(skin)));
        meshes[`${side}_hand`] = handG;

        const thighG = new THREE.Group();
        thighG.add(cap(thigh, 0.07, hMat(cloth)));
        meshes[`${side}_thigh`] = thighG;

        const shinG = new THREE.Group();
        shinG.add(cap(shin, 0.05, hMat(skin)));
        meshes[`${side}_shin`] = shinG;

        const footG = new THREE.Group();
        footG.add(new THREE.Mesh(new THREE.BoxGeometry(0.10, 0.20, h * 0.04), hMat(0x111111)));
        meshes[`${side}_foot`] = footG;
    }

    return meshes;
}

// ─── Wall + holds ─────────────────────────────────────────────────────────
function buildWallAndHolds(wall, pose) {
    // Wall as a thin rectangle. We use the wall extents from the
    // session response. Apply the wall_angle as a rotation about X.
    const w = wall.width_m;
    const h = wall.height_m;
    const theta = wall.wall_angle_deg * Math.PI / 180;

    const pad = 0.30;
    const plate = new THREE.Mesh(
        new THREE.BoxGeometry(w + pad * 2, 0.05, h + pad * 2),
        new THREE.MeshStandardMaterial({ color: 0xd9d2c4, roughness: 0.85 }),
    );
    plate.castShadow = false;
    plate.receiveShadow = true;
    // Bottom edge at z = 0, plate rotated by +theta around X so top
    // tilts toward +Y for overhang. Centre at (0, h/2 sin θ, h/2 cos θ).
    plate.position.set(0, (h / 2) * Math.sin(theta), (h / 2) * Math.cos(theta));
    plate.rotation.x = theta;
    scene.add(plate);

    // Holds: we already have world positions from the pose snapshot.
    holdMeshes = {};
    holdRings = {};
    for (const [hid, pos] of Object.entries(pose.holds)) {
        const radius = 0.045;       // visual default — actual radius comes from size
        const cyl = new THREE.Mesh(
            new THREE.CylinderGeometry(radius, radius, 0.04, 16),
            new THREE.MeshStandardMaterial({ color: 0x888888, roughness: 0.5 }),
        );
        cyl.position.set(pos[0], pos[1], pos[2]);
        // Orient cylinder along the wall normal (approximately +Y for
        // vertical wall; for slab/overhang we'd rotate, but the visual
        // is forgiving).
        cyl.rotation.x = Math.PI / 2;
        cyl.rotation.z = theta;
        cyl.castShadow = true;
        scene.add(cyl);
        holdMeshes[hid] = cyl;
    }

    return plate;
}

// ─── Pose update ──────────────────────────────────────────────────────────
function quatThreeFromMujoco(q) {
    // MuJoCo quat layout = [w, x, y, z]; three.js quaternion = (x, y, z, w).
    return new THREE.Quaternion(q[1], q[2], q[3], q[0]);
}

function applyPose(pose) {
    for (const [name, mesh] of Object.entries(bodyMeshes)) {
        const data = pose.bodies[name];
        if (!data) continue;
        if (mesh.parent !== scene) scene.add(mesh);
        mesh.position.set(data.pos[0], data.pos[1], data.pos[2]);
        mesh.quaternion.copy(quatThreeFromMujoco(data.quat));
    }

    // Highlight holds the climber is on.
    const onHolds = new Set(Object.values(pose.limbs).filter(Boolean));
    for (const [hid, mesh] of Object.entries(holdMeshes)) {
        if (onHolds.has(hid)) {
            mesh.material.emissive = new THREE.Color(0x445500);
        } else {
            mesh.material.emissive = new THREE.Color(0x000000);
        }
    }

    // Update status panel
    const com = pose.bodies.pelvis ? pose.bodies.pelvis.pos : [0, 0, 0];
    const limbStr = LIMBS.map(l => `${l}: ${pose.limbs[l] ?? '·'}`).join('\n');
    document.getElementById('status').textContent =
        `t = ${pose.t.toFixed(2)} s\npelvis = (${com.map(v => v.toFixed(2)).join(', ')})\n${limbStr}`;
}

// ─── Wall list ────────────────────────────────────────────────────────────
async function loadWallList() {
    const sel = document.getElementById('wall-select');
    const r = await fetch('/api/walls');
    const j = await r.json();
    sel.innerHTML = '';
    for (const wid of j.walls) {
        const opt = document.createElement('option');
        opt.value = wid;
        opt.textContent = wid;
        sel.appendChild(opt);
    }
    if (j.walls.length === 0) {
        sel.innerHTML = '<option>(no walls — create one in the editor)</option>';
    }
}

// ─── Session control ──────────────────────────────────────────────────────
async function startSession() {
    if (session) {
        await fetch(`/sim3d/api/session/${session.id}`, { method: 'DELETE' });
        // Tear down old meshes
        for (const m of Object.values(bodyMeshes)) scene.remove(m);
        for (const m of Object.values(holdMeshes)) scene.remove(m);
        bodyMeshes = {};
        holdMeshes = {};
    }

    const payload = {
        wall_id: document.getElementById('wall-select').value,
        height_cm: parseFloat(document.getElementById('height').value),
        wingspan_cm: parseFloat(document.getElementById('wingspan').value),
        mass_kg: parseFloat(document.getElementById('mass').value),
        seed: true,
    };
    const r = await fetch('/sim3d/api/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    if (!r.ok) {
        document.getElementById('status').textContent = `Failed: ${await r.text()}`;
        return;
    }
    const j = await r.json();
    session = { id: j.session_id, wall: j.wall, profile: j.profile, pose: j.pose };

    bodyMeshes = buildClimberMeshes(j.profile);
    for (const m of Object.values(bodyMeshes)) scene.add(m);
    buildWallAndHolds(j.wall, j.pose);
    applyPose(j.pose);

    // Centre the camera target on the wall mid-height.
    controls.target.set(0, 0, j.wall.height_m / 2);
    controls.update();

    populateLimbControls(j.pose);
    document.getElementById('reset-btn').disabled = false;
}

function populateLimbControls(pose) {
    const ctrls = document.getElementById('limb-controls');
    ctrls.innerHTML = '';
    const holdIds = Object.keys(pose.holds).sort();
    for (const limb of LIMBS) {
        const row = document.createElement('div');
        row.className = 'limb-row';
        row.innerHTML = `<span>${limb}</span>`;
        const sel = document.createElement('select');
        for (const hid of holdIds) {
            const opt = document.createElement('option');
            opt.value = hid;
            opt.textContent = hid;
            if (pose.limbs[limb] === hid) opt.selected = true;
            sel.appendChild(opt);
        }
        sel.onchange = async () => {
            const r = await fetch(`/sim3d/api/session/${session.id}/move`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ limb, hold_id: sel.value, mode: 'reach' }),
            });
            if (r.ok) applyPose(await r.json());
        };
        row.appendChild(sel);
        ctrls.appendChild(row);
    }
}

async function step(frames) {
    if (!session) return;
    const r = await fetch(`/sim3d/api/session/${session.id}/step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frames }),
    });
    if (r.ok) applyPose(await r.json());
}

document.getElementById('start-btn').onclick = startSession;
document.getElementById('step1').onclick = () => step(6);    // 0.1 s
document.getElementById('step10').onclick = () => step(60);  // 1.0 s
document.getElementById('reset-btn').onclick = async () => {
    if (!session) return;
    const r = await fetch(`/sim3d/api/session/${session.id}/seed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
    });
    if (r.ok) applyPose(await r.json());
};

const playBtn = document.getElementById('play');
playBtn.onclick = () => {
    if (liveTimer) {
        clearInterval(liveTimer);
        liveTimer = null;
        playBtn.textContent = '▶ Live (60 Hz)';
        return;
    }
    playBtn.textContent = '⏸ Pause';
    // 6 frames per request, ~10 requests/sec → ~real-time at 60 Hz sim.
    liveTimer = setInterval(() => step(6), 100);
};

// ─── Boot ─────────────────────────────────────────────────────────────────
loadWallList();
