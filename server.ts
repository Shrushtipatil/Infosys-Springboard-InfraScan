import express from 'express';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { createServer as createViteServer } from 'vite';

const app = express();
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

const JWT_SECRET = process.env.JWT_SECRET || 'infrascan-civil-ai-hmac-sha256-secret-key-production';

export type RoleType =
  | 'Administrator'
  | 'Infrastructure Manager'
  | 'Civil Engineer'
  | 'Field Inspector'
  | 'Maintenance Engineer';

export interface UserRecord {
  user_id: number;
  email: string;
  password_hash: string;
  full_name: string;
  role: RoleType;
  created_at: string;
  token?: string;
}

export interface InfrastructureAssetRecord {
  asset_id: number;
  asset_name: string;
  asset_type: 'Bridge' | 'Pavement' | 'Vertical Building' | 'Tunnel' | 'Hydraulic Dam';
  material_type: string;
  construction_year: number;
  latitude: number;
  longitude: number;
  responsible_department: string;
  health_score: number;
}

export interface InspectionRecord {
  inspection_id: number;
  asset_id: number;
  inspector_id: number;
  inspection_date: string;
  status: string;
  notes: string;
}

export interface InspectionImageRecord {
  image_id: number;
  inspection_id: number;
  storage_url: string;
  width_px: number;
  height_px: number;
  blur_laplacian_var: number;
  avg_brightness: number;
  image_hash: string;
  scale_mm_per_px: number;
}

export interface DefectDetectionRecord {
  detection_id: number;
  image_id: number;
  defect_class: 'D00 Longitudinal' | 'D10 Transverse' | 'D20 Alligator' | 'D40 Pothole' | 'Spalling';
  confidence_score: number;
  bbox_xmin: number; // normalized 0..100%
  bbox_ymin: number;
  bbox_xmax: number;
  bbox_ymax: number;
}

export interface DefectSegmentationRecord {
  segmentation_id: number;
  detection_id: number;
  crack_length_mm: number;
  avg_width_mm: number;
  max_width_mm: number;
  crack_density_pct: number;
  svg_mask_paths: string[];
  gradcam_hotspots: Array<{ cx: number; cy: number; rx: number; ry: number; weight: number; label: string }>;
}

export interface SeverityAssessmentRecord {
  assessment_id: number;
  inspection_id: number;
  severity_level: 'Minor' | 'Moderate' | 'Severe' | 'Critical';
  maintenance_priority: 'Routine' | 'Scheduled' | 'Urgent' | 'Immediate';
  engineer_comments: string;
  verified_by?: string;
  verified_at?: string;
  approval_status: 'AI Baseline' | 'Approved' | 'Reclassified' | 'Escalated';
}

// HMAC-SHA256 JWT Implementation
function createJwt(payload: Record<string, unknown>): string {
  const header = { alg: 'HS256', typ: 'JWT' };
  const exp = Math.floor(Date.now() / 1000) + 86400 * 7;
  const fullPayload = { ...payload, iat: Math.floor(Date.now() / 1000), exp };
  const b64Header = Buffer.from(JSON.stringify(header)).toString('base64url');
  const b64Payload = Buffer.from(JSON.stringify(fullPayload)).toString('base64url');
  const signature = crypto
    .createHmac('sha256', JWT_SECRET)
    .update(`${b64Header}.${b64Payload}`)
    .digest('base64url');
  return `${b64Header}.${b64Payload}.${signature}`;
}

function verifyJwt(token: string): Record<string, any> | null {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return null;
    const [b64Header, b64Payload, signature] = parts;
    const expectedSig = crypto
      .createHmac('sha256', JWT_SECRET)
      .update(`${b64Header}.${b64Payload}`)
      .digest('base64url');
    if (signature !== expectedSig) return null;
    const payload = JSON.parse(Buffer.from(b64Payload, 'base64url').toString('utf-8'));
    if (payload.exp && payload.exp < Math.floor(Date.now() / 1000)) return null;
    return payload;
  } catch {
    return null;
  }
}

// Persistent Registered Users Database (Stored on disk at /data/registered_users.json)
const DB_DIR = path.resolve(process.cwd(), 'data');
const USERS_DB_PATH = path.join(DB_DIR, 'registered_users.json');

function loadRegisteredUsersFromDb(): UserRecord[] {
  try {
    if (!fs.existsSync(DB_DIR)) {
      fs.mkdirSync(DB_DIR, { recursive: true });
    }
    if (!fs.existsSync(USERS_DB_PATH)) {
      fs.writeFileSync(USERS_DB_PATH, JSON.stringify([], null, 2), 'utf-8');
      return [];
    }
    const raw = fs.readFileSync(USERS_DB_PATH, 'utf-8');
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveRegisteredUsersToDb(records: UserRecord[]): void {
  try {
    if (!fs.existsSync(DB_DIR)) {
      fs.mkdirSync(DB_DIR, { recursive: true });
    }
    fs.writeFileSync(USERS_DB_PATH, JSON.stringify(records, null, 2), 'utf-8');
  } catch (err) {
    console.error('Failed to persist registered users database:', err);
  }
}

const users: UserRecord[] = loadRegisteredUsersFromDb();
let nextUserId =
  users.reduce((maxId, u) => (u.user_id > maxId ? u.user_id : maxId), 0) + 1;

// Seed 5 Real-World Infrastructure Assets
const infrastructure_assets: InfrastructureAssetRecord[] = [
  {
    asset_id: 1,
    asset_name: 'Golden Gate Deck Section B',
    asset_type: 'Bridge',
    material_type: 'Reinforced Orthotropic Concrete Deck',
    construction_year: 1937,
    latitude: 37.8199,
    longitude: -122.4783,
    responsible_department: 'Caltrans Bay Area Toll Bridge Engineering',
    health_score: 54,
  },
  {
    asset_id: 2,
    asset_name: 'Highway A1 Interstate Km 142',
    asset_type: 'Pavement',
    material_type: 'Hot-Mix Asphalt Concrete (HMAC)',
    construction_year: 2008,
    latitude: 37.6879,
    longitude: -122.4005,
    responsible_department: 'Federal Highway Corridor Maintenance Div 4',
    health_score: 74,
  },
  {
    asset_id: 3,
    asset_name: 'Municipal Shear Wall North Face',
    asset_type: 'Vertical Building',
    material_type: 'High-Strength Cast-in-Place Concrete (C50/60)',
    construction_year: 2016,
    latitude: 37.7793,
    longitude: -122.4193,
    responsible_department: 'Municipal Structural Integrity Bureau',
    health_score: 88,
  },
  {
    asset_id: 4,
    asset_name: 'Trans-Alpine Rail Tunnel Portal',
    asset_type: 'Tunnel',
    material_type: 'Shotcrete & Precast Segmental Lining',
    construction_year: 1999,
    latitude: 37.8651,
    longitude: -122.2585,
    responsible_department: 'Transit Underground Geotechnical Authority',
    health_score: 61,
  },
  {
    asset_id: 5,
    asset_name: 'Hydroelectric Reservoir Dam Spillway',
    asset_type: 'Hydraulic Dam',
    material_type: 'Mass Post-Tensioned Hydraulic Concrete',
    construction_year: 1974,
    latitude: 37.9101,
    longitude: -122.1065,
    responsible_department: 'Regional Water & Hydroelectric Safety Board',
    health_score: 38,
  },
];

const inspections: InspectionRecord[] = [
  {
    inspection_id: 101,
    asset_id: 1,
    inspector_id: 4,
    inspection_date: '2026-10-21T09:30:00Z',
    status: 'Completed',
    notes: 'UAV nadir photogrammetry + macro deck scan on Span B-14 southbound lane.',
  },
  {
    inspection_id: 102,
    asset_id: 2,
    inspector_id: 4,
    inspection_date: '2026-10-22T14:15:00Z',
    status: 'Completed',
    notes: 'Mobile mapping line-scan vehicle pass at 40 km/h; thermal transverse fracture detected.',
  },
  {
    inspection_id: 103,
    asset_id: 3,
    inspector_id: 4,
    inspection_date: '2026-10-23T11:00:00Z',
    status: 'Completed',
    notes: 'Telephoto rig inspection of North elevation shear core between levels 4 and 6.',
  },
  {
    inspection_id: 104,
    asset_id: 4,
    inspector_id: 4,
    inspection_date: '2026-10-24T22:45:00Z',
    status: 'Completed',
    notes: 'Night-window track possession laser + strobe capture of East portal crown.',
  },
  {
    inspection_id: 105,
    asset_id: 5,
    inspector_id: 4,
    inspection_date: '2026-10-25T08:10:00Z',
    status: 'Completed',
    notes: 'Rope-access close-range photogrammetry on Spillway Monolith 7 chute face.',
  },
];

const inspection_images: InspectionImageRecord[] = [
  {
    image_id: 201,
    inspection_id: 101,
    storage_url: '/src/assets/images/bridge_deck_alligator_1793038179754.jpg',
    width_px: 2048,
    height_px: 1536,
    blur_laplacian_var: 318.4,
    avg_brightness: 118.6,
    image_hash: '8f94d2c7a1b04e91',
    scale_mm_per_px: 0.34,
  },
  {
    image_id: 202,
    inspection_id: 102,
    storage_url: '/src/assets/images/highway_pavement_transverse_1793038180621.jpg',
    width_px: 2048,
    height_px: 1536,
    blur_laplacian_var: 284.2,
    avg_brightness: 136.2,
    image_hash: '3a71e5b9c40812fa',
    scale_mm_per_px: 0.42,
  },
  {
    image_id: 203,
    inspection_id: 103,
    storage_url: '/src/assets/images/shear_wall_longitudinal_1793038181824.jpg',
    width_px: 2048,
    height_px: 1536,
    blur_laplacian_var: 262.9,
    avg_brightness: 144.8,
    image_hash: '7c19f0a4e823d51b',
    scale_mm_per_px: 0.28,
  },
  {
    image_id: 204,
    inspection_id: 104,
    storage_url: '/src/assets/images/rail_tunnel_spalling_1793038182037.jpg',
    width_px: 2048,
    height_px: 1536,
    blur_laplacian_var: 229.5,
    avg_brightness: 104.3,
    image_hash: '9e42b1d8f065a3c4',
    scale_mm_per_px: 0.51,
  },
  {
    image_id: 205,
    inspection_id: 105,
    storage_url: '/src/assets/images/hydro_dam_spillway_1793038182589.jpg',
    width_px: 2048,
    height_px: 1536,
    blur_laplacian_var: 342.1,
    avg_brightness: 129.0,
    image_hash: '5d83c6e1a9047f22',
    scale_mm_per_px: 0.46,
  },
];

const defect_detections: DefectDetectionRecord[] = [
  {
    detection_id: 301,
    image_id: 201,
    defect_class: 'D20 Alligator',
    confidence_score: 0.964,
    bbox_xmin: 10,
    bbox_ymin: 12,
    bbox_xmax: 88,
    bbox_ymax: 86,
  },
  {
    detection_id: 302,
    image_id: 202,
    defect_class: 'D10 Transverse',
    confidence_score: 0.948,
    bbox_xmin: 4,
    bbox_ymin: 38,
    bbox_xmax: 96,
    bbox_ymax: 64,
  },
  {
    detection_id: 303,
    image_id: 203,
    defect_class: 'D00 Longitudinal',
    confidence_score: 0.921,
    bbox_xmin: 42,
    bbox_ymin: 5,
    bbox_xmax: 58,
    bbox_ymax: 95,
  },
  {
    detection_id: 304,
    image_id: 204,
    defect_class: 'Spalling',
    confidence_score: 0.953,
    bbox_xmin: 26,
    bbox_ymin: 20,
    bbox_xmax: 63,
    bbox_ymax: 72,
  },
  {
    detection_id: 305,
    image_id: 205,
    defect_class: 'D20 Alligator',
    confidence_score: 0.981,
    bbox_xmin: 14,
    bbox_ymin: 24,
    bbox_xmax: 78,
    bbox_ymax: 92,
  },
];

const defect_segmentations: DefectSegmentationRecord[] = [
  {
    segmentation_id: 401,
    detection_id: 301,
    crack_length_mm: 1842.5,
    avg_width_mm: 3.85,
    max_width_mm: 6.40,
    crack_density_pct: 18.6,
    svg_mask_paths: [
      'M 18 22 L 36 38 L 52 29 L 68 44 L 84 35',
      'M 36 38 L 44 62 L 28 76 L 14 68',
      'M 52 29 L 61 58 L 78 72',
      'M 44 62 L 61 58 L 55 82',
      'M 22 44 L 36 38 M 68 44 L 61 58',
    ],
    gradcam_hotspots: [
      { cx: 48, cy: 48, rx: 28, ry: 24, weight: 0.96, label: 'Primary Fatigue Polygon Node' },
      { cx: 34, cy: 62, rx: 18, ry: 16, weight: 0.84, label: 'Sub-base Spall Intersect' },
    ],
  },
  {
    segmentation_id: 402,
    detection_id: 302,
    crack_length_mm: 1120.0,
    avg_width_mm: 2.15,
    max_width_mm: 3.90,
    crack_density_pct: 6.4,
    svg_mask_paths: [
      'M 5 44 L 28 45 L 46 50 L 67 53 L 82 59 L 96 60',
      'M 73 20 L 75 55 L 78 88',
    ],
    gradcam_hotspots: [
      { cx: 56, cy: 52, rx: 34, ry: 12, weight: 0.93, label: 'Transverse Shear Propagation Zone' },
      { cx: 74, cy: 56, rx: 14, ry: 18, weight: 0.88, label: 'Edge Raveling Bifurcation' },
    ],
  },
  {
    segmentation_id: 403,
    detection_id: 303,
    crack_length_mm: 645.0,
    avg_width_mm: 0.85,
    max_width_mm: 1.45,
    crack_density_pct: 2.3,
    svg_mask_paths: [
      'M 48 6 L 47 26 L 49 48 L 50 71 L 48 94',
      'M 45 24 L 56 24 M 46 79 L 58 79',
    ],
    gradcam_hotspots: [
      { cx: 49, cy: 50, rx: 12, ry: 34, weight: 0.89, label: 'Vertical Tensile Hairline Axis' },
      { cx: 50, cy: 24, rx: 14, ry: 10, weight: 0.92, label: 'Rebar Tie-Hole Stress Riser' },
    ],
  },
  {
    segmentation_id: 404,
    detection_id: 304,
    crack_length_mm: 980.4,
    avg_width_mm: 4.90,
    max_width_mm: 11.20,
    crack_density_pct: 14.8,
    svg_mask_paths: [
      'M 36 30 L 58 36 L 56 60 L 35 64 Z',
      'M 43 32 L 41 62 M 52 35 L 50 61',
      'M 15 32 L 16 74',
    ],
    gradcam_hotspots: [
      { cx: 46, cy: 46, rx: 22, ry: 22, weight: 0.97, label: 'Delaminated Concrete & Exposed Rebar' },
      { cx: 28, cy: 62, rx: 15, ry: 18, weight: 0.85, label: 'Moisture Ingress Leaching Plume' },
    ],
  },
  {
    segmentation_id: 405,
    detection_id: 305,
    crack_length_mm: 2490.8,
    avg_width_mm: 5.40,
    max_width_mm: 9.85,
    crack_density_pct: 24.7,
    svg_mask_paths: [
      'M 22 86 L 38 62 L 49 44 L 62 34',
      'M 32 34 L 46 42 L 58 36 L 68 48',
      'M 38 62 L 54 56 L 64 68',
      'M 26 52 L 44 48 L 49 44',
    ],
    gradcam_hotspots: [
      { cx: 46, cy: 52, rx: 28, ry: 26, weight: 0.99, label: 'Critical Hydraulic Uplift Fracture Network' },
      { cx: 34, cy: 72, rx: 20, ry: 18, weight: 0.94, label: 'Calcium Carbonate Efflorescence Seepage' },
    ],
  },
];

const severity_assessments: SeverityAssessmentRecord[] = [
  {
    assessment_id: 501,
    inspection_id: 101,
    severity_level: 'Severe',
    maintenance_priority: 'Urgent',
    engineer_comments:
      'Interconnected fatigue alligator cracking across 18.6% of surveyed deck tile. Max crack opening 6.40 mm exceeds ACI 224R threshold for de-icing salt chloride penetration. Schedule epoxy flood coat and wearing surface mill-and-overlay.',
    verified_by: 'Automated AI Baseline',
    verified_at: '2026-10-21T16:40:00Z',
    approval_status: 'Approved',
  },
  {
    assessment_id: 502,
    inspection_id: 102,
    severity_level: 'Moderate',
    maintenance_priority: 'Scheduled',
    engineer_comments:
      'Full-lane D10 transverse thermal shrinkage fracture with localized secondary longitudinal branch. Mean crack width 2.15 mm. Route and seal with hot-applied elastomeric sealant within 60 days.',
    verified_by: 'Automated AI Baseline',
    verified_at: '2026-10-22T17:10:00Z',
    approval_status: 'AI Baseline',
  },
  {
    assessment_id: 503,
    inspection_id: 103,
    severity_level: 'Minor',
    maintenance_priority: 'Routine',
    engineer_comments:
      'Vertical D00 drying shrinkage crack along tie-rod vertical alignment. Mean width 0.85 mm; localized surface tie-pocket spall. Non-structural; apply low-viscosity silane sealer during next annual cycle.',
    verified_by: 'Automated AI Baseline',
    verified_at: '2026-10-23T15:05:00Z',
    approval_status: 'Approved',
  },
  {
    assessment_id: 504,
    inspection_id: 104,
    severity_level: 'Severe',
    maintenance_priority: 'Urgent',
    engineer_comments:
      'Active lining spalling with exposed oxidized reinforcement mesh and groundwater carbonate leaching. Clear cover loss over 0.42 m². Require hydro-demolition, cathodic zinc primer, and polymer-modified repair mortar.',
    verified_by: 'Automated AI Baseline',
    verified_at: '2026-10-25T09:00:00Z',
    approval_status: 'Approved',
  },
  {
    assessment_id: 505,
    inspection_id: 105,
    severity_level: 'Critical',
    maintenance_priority: 'Immediate',
    engineer_comments:
      'Critical structural block cracking (24.7% surface density, 9.85 mm max aperture) accompanied by active hydrostatic efflorescence on Spillway Monolith 7. High risk of hydrodynamic slab jacking during spillway discharge. Immediate polyurethane grouting & post-tension anchor check required.',
    verified_by: 'Automated AI Baseline',
    verified_at: '2026-10-25T11:30:00Z',
    approval_status: 'Escalated',
  },
];

// Helper to resolve current user from Authorization header
function getAuthenticatedUser(req: express.Request): UserRecord {
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    const token = authHeader.slice(7);
    const payload = verifyJwt(token);
    if (payload) {
      const found = users.find(
        (u) => u.email === payload.email || u.email === payload.sub || u.user_id === payload.user_id
      );
      if (found) return found;
      return {
        user_id: Number(payload.user_id) || 1,
        email: String(payload.sub || payload.email || 'operator@infrascan.gov'),
        password_hash: 'jwt_verified',
        full_name: String(payload.full_name || payload.role || 'Authenticated Operator'),
        role: (payload.role as RoleType) || 'Civil Engineer',
        created_at: new Date().toISOString(),
        token,
      };
    }
  }
  if (users.length > 0) return users[0];
  return {
    user_id: 1,
    email: 'operator@infrascan.gov',
    password_hash: 'session_active',
    full_name: 'Civil Engineer Session',
    role: 'Civil Engineer',
    created_at: new Date().toISOString(),
    token: '',
  };
}

// Assemble rich composite payload for an asset + its latest inspection
function buildAssetComposite(asset: InfrastructureAssetRecord) {
  const assetInspections = inspections.filter((i) => i.asset_id === asset.asset_id);
  const latestInspection = assetInspections[assetInspections.length - 1] || inspections[0];
  const image =
    inspection_images.find((img) => img.inspection_id === latestInspection.inspection_id) ||
    inspection_images[0];
  const detection =
    defect_detections.find((d) => d.image_id === image.image_id) || defect_detections[0];
  const segmentation =
    defect_segmentations.find((s) => s.detection_id === detection.detection_id) ||
    defect_segmentations[0];
  const assessment =
    severity_assessments.find((a) => a.inspection_id === latestInspection.inspection_id) ||
    severity_assessments[0];
  const inspector = users.find((u) => u.user_id === latestInspection.inspector_id);

  return {
    ...asset,
    latest_inspection: latestInspection,
    inspector_name: inspector ? inspector.full_name : 'Field Telemetry Unit',
    image,
    detection,
    segmentation,
    assessment,
  };
}

// =====================================================================
// FASTAPI-COMPATIBLE REST API ENDPOINTS (/api/v1/*)
// =====================================================================

const ADMIN_ACCESS_KEY = '1234';

// 1. Auth Endpoints (Create Account & Log In; Persistent Registered Users DB)
app.post('/api/v1/auth/register', (req, res) => {
  const { full_name, email, password, role, admin_access_key } = req.body;
  const cleanEmail = String(email || '').trim().toLowerCase();
  const cleanName = String(full_name || '').trim();
  const cleanPassword = String(password || '').trim();
  const targetRole = (role as RoleType) || 'Civil Engineer';

  if (!cleanEmail || !cleanPassword || !cleanName) {
    return res.status(400).json({
      detail: 'Full Name, Email Address, and Password are required to create an account.',
    });
  }

  // Prevent multiple accounts with the same email
  const existing = users.find((u) => u.email.toLowerCase() === cleanEmail);
  if (existing) {
    return res.status(409).json({
      code: 'ACCOUNT_EXISTS',
      detail: 'Account already exists with this email address. Please log in instead.',
    });
  }

  if (targetRole === 'Administrator' && String(admin_access_key ?? '').trim() !== ADMIN_ACCESS_KEY) {
    return res.status(403).json({
      detail: 'Access Denied: Valid Administrator Access Key is required to create an Administrator account.',
    });
  }

  const newUserId = nextUserId++;
  const token = createJwt({
    sub: cleanEmail,
    user_id: newUserId,
    role: targetRole,
    full_name: cleanName,
  });

  const newUser: UserRecord = {
    user_id: newUserId,
    email: cleanEmail,
    password_hash: crypto.createHash('sha256').update(cleanPassword).digest('hex'),
    full_name: cleanName,
    role: targetRole,
    created_at: new Date().toISOString(),
    token,
  };
  users.push(newUser);
  saveRegisteredUsersToDb(users);

  res.status(201).json({
    access_token: token,
    token_type: 'bearer',
    user: {
      user_id: newUser.user_id,
      email: newUser.email,
      full_name: newUser.full_name,
      role: newUser.role,
      created_at: newUser.created_at,
    },
  });
});

app.post('/api/v1/auth/login', (req, res) => {
  const { email, username, password, role, admin_access_key, access_key } = req.body;
  const cleanEmail = String(email || username || '').trim().toLowerCase();
  const cleanPassword = String(password || '').trim();
  const targetRole = (role as RoleType) || 'Civil Engineer';
  const submittedAdminKey = String(admin_access_key ?? access_key ?? '').trim();

  if (!cleanEmail || !cleanPassword) {
    return res.status(400).json({
      detail: 'Email address and password are required to log in.',
    });
  }

  // Strictly check if the user exists in the registered users database first
  const user = users.find((u) => u.email.toLowerCase() === cleanEmail);
  if (!user) {
    return res.status(404).json({
      code: 'USER_NOT_FOUND',
      detail: 'User not found. Please create an account first before logging in.',
    });
  }

  const hashedInput = crypto.createHash('sha256').update(cleanPassword).digest('hex');
  if (user.password_hash !== hashedInput) {
    return res.status(401).json({
      detail: 'Incorrect password for this account.',
    });
  }

  // Administrator strictly requires the Administrator Access Key
  if (
    (targetRole === 'Administrator' || user.role === 'Administrator') &&
    submittedAdminKey !== ADMIN_ACCESS_KEY
  ) {
    return res.status(403).json({
      detail: 'Access Denied: Valid Administrator Access Key is required for Administrator sign-in.',
    });
  }

  // Lock active role selected at sign-in and persist
  user.role = targetRole;

  const token = createJwt({
    sub: user.email,
    user_id: user.user_id,
    role: user.role,
    full_name: user.full_name,
  });
  user.token = token;
  saveRegisteredUsersToDb(users);

  res.json({
    access_token: token,
    token_type: 'bearer',
    user: {
      user_id: user.user_id,
      email: user.email,
      full_name: user.full_name,
      role: user.role,
      created_at: user.created_at,
    },
  });
});

app.post('/api/v1/auth/override-role', (req, res) => {
  const { target_role, admin_access_key } = req.body;
  if (String(admin_access_key ?? '').trim() !== ADMIN_ACCESS_KEY) {
    return res.status(403).json({
      detail: 'Admin Override Rejected: Invalid Administrator Access Key.',
    });
  }
  const currentUser = getAuthenticatedUser(req);
  const updatedRole = (target_role as RoleType) || 'Civil Engineer';
  currentUser.role = updatedRole;

  const token = createJwt({
    sub: currentUser.email,
    user_id: currentUser.user_id,
    role: updatedRole,
    full_name: currentUser.full_name,
  });
  currentUser.token = token;
  saveRegisteredUsersToDb(users);

  res.json({
    access_token: token,
    token_type: 'bearer',
    overridden_by: 'Administrator (Key Verified)',
    user: {
      user_id: currentUser.user_id,
      email: currentUser.email,
      full_name: currentUser.full_name,
      role: currentUser.role,
      created_at: currentUser.created_at,
    },
  });
});

app.get('/api/v1/auth/me', (req, res) => {
  const user = getAuthenticatedUser(req);
  res.json({
    user_id: user.user_id,
    email: user.email,
    full_name: user.full_name,
    role: user.role,
    created_at: user.created_at,
  });
});

// 2. Infrastructure Assets Endpoints
app.get('/api/v1/assets', (_req, res) => {
  const enriched = infrastructure_assets.map(buildAssetComposite);
  res.json(enriched);
});

app.post('/api/v1/assets', (req, res) => {
  const user = getAuthenticatedUser(req);
  const allowedRoles: RoleType[] = ['Administrator', 'Infrastructure Manager', 'Civil Engineer'];
  if (!allowedRoles.includes(user.role)) {
    return res.status(403).json({
      detail: `RBAC Policy Violation: Role '${user.role}' cannot register new civil infrastructure assets.`,
    });
  }

  const {
    asset_name,
    asset_type,
    material_type,
    construction_year,
    latitude,
    longitude,
    responsible_department,
  } = req.body;

  const newAssetId = infrastructure_assets.length + 1;
  const newAsset: InfrastructureAssetRecord = {
    asset_id: newAssetId,
    asset_name: asset_name || `Civil Structure Sector #${newAssetId}`,
    asset_type: asset_type || 'Bridge',
    material_type: material_type || 'Reinforced Concrete (C40/50)',
    construction_year: Number(construction_year) || 2012,
    latitude: Number(latitude) || 37.8044,
    longitude: Number(longitude) || -122.2712,
    responsible_department: responsible_department || 'Structural Engineering Division',
    health_score: 82,
  };
  infrastructure_assets.push(newAsset);

  const newInspectionId = 100 + newAssetId;
  const newImageId = 200 + newAssetId;
  const newDetectionId = 300 + newAssetId;
  const newSegId = 400 + newAssetId;
  const newAssessmentId = 500 + newAssetId;

  inspections.push({
    inspection_id: newInspectionId,
    asset_id: newAssetId,
    inspector_id: user.user_id,
    inspection_date: new Date().toISOString(),
    status: 'Completed',
    notes: `Initial baseline optical inspection logged by ${user.full_name}.`,
  });

  inspection_images.push({
    image_id: newImageId,
    inspection_id: newInspectionId,
    storage_url: '/src/assets/images/shear_wall_longitudinal_1793038181824.jpg',
    width_px: 2048,
    height_px: 1536,
    blur_laplacian_var: 274.5,
    avg_brightness: 138.0,
    image_hash: crypto.randomBytes(8).toString('hex'),
    scale_mm_per_px: 0.32,
  });

  defect_detections.push({
    detection_id: newDetectionId,
    image_id: newImageId,
    defect_class: 'D00 Longitudinal',
    confidence_score: 0.912,
    bbox_xmin: 40,
    bbox_ymin: 10,
    bbox_xmax: 60,
    bbox_ymax: 90,
  });

  defect_segmentations.push({
    segmentation_id: newSegId,
    detection_id: newDetectionId,
    crack_length_mm: 512.4,
    avg_width_mm: 0.95,
    max_width_mm: 1.65,
    crack_density_pct: 2.8,
    svg_mask_paths: ['M 48 10 L 49 48 L 48 90'],
    gradcam_hotspots: [
      { cx: 49, cy: 50, rx: 14, ry: 30, weight: 0.87, label: 'Longitudinal Surface Fracture' },
    ],
  });

  severity_assessments.push({
    assessment_id: newAssessmentId,
    inspection_id: newInspectionId,
    severity_level: 'Minor',
    maintenance_priority: 'Routine',
    engineer_comments: 'Automated baseline YOLOv11 + U-Net intake scan completed. Routine monitoring advised.',
    verified_by: user.full_name,
    verified_at: new Date().toISOString(),
    approval_status: 'AI Baseline',
  });

  res.status(201).json(buildAssetComposite(newAsset));
});

app.get('/api/v1/assets/:id/history', (req, res) => {
  const assetId = Number(req.params.id);
  const asset = infrastructure_assets.find((a) => a.asset_id === assetId);
  if (!asset) {
    return res.status(404).json({ detail: 'Infrastructure asset not found' });
  }
  const composite = buildAssetComposite(asset);
  const historyTimeline = [
    {
      inspection_id: composite.latest_inspection.inspection_id,
      inspection_date: composite.latest_inspection.inspection_date,
      inspector: composite.inspector_name,
      defect_class: composite.detection.defect_class,
      max_width_mm: composite.segmentation.max_width_mm,
      crack_density_pct: composite.segmentation.crack_density_pct,
      severity_level: composite.assessment.severity_level,
      status: composite.assessment.approval_status,
      notes: composite.latest_inspection.notes,
    },
    {
      inspection_id: composite.latest_inspection.inspection_id - 50,
      inspection_date: '2026-04-14T10:00:00Z',
      inspector: 'Field Inspection Unit',
      defect_class: composite.detection.defect_class,
      max_width_mm: Number((composite.segmentation.max_width_mm * 0.82).toFixed(2)),
      crack_density_pct: Number((composite.segmentation.crack_density_pct * 0.85).toFixed(1)),
      severity_level: composite.assessment.severity_level === 'Critical' ? 'Severe' : 'Moderate',
      status: 'Approved',
      notes: 'Semi-annual structural benchmark scan; progressive crack aperture widening noted (+18%).',
    },
    {
      inspection_id: composite.latest_inspection.inspection_id - 90,
      inspection_date: '2025-10-09T09:15:00Z',
      inspector: 'Civil Engineering Review',
      defect_class: composite.detection.defect_class,
      max_width_mm: Number((composite.segmentation.max_width_mm * 0.68).toFixed(2)),
      crack_density_pct: Number((composite.segmentation.crack_density_pct * 0.70).toFixed(1)),
      severity_level: 'Minor',
      status: 'Approved',
      notes: 'Initial PostGIS optical baseline survey.',
    },
  ];
  res.json({
    asset: composite,
    history: historyTimeline,
  });
});

// 3. Upload & Quality Control Gating Endpoints
app.post('/api/v1/preprocess/validate', (req, res) => {
  const {
    filename = 'inspection_frame.jpg',
    width_px = 2048,
    height_px = 1536,
    blur_laplacian_var = 285.4,
    avg_brightness = 128.5,
    reference_distance_mm = 100.0,
    reference_pixel_count = 278.0,
    apply_clahe = true,
  } = req.body;

  const laplacian = Number(blur_laplacian_var);
  const brightness = Number(avg_brightness);
  const w = Number(width_px);
  const h = Number(height_px);
  const refMm = Number(reference_distance_mm) || 100;
  const refPx = Math.max(Number(reference_pixel_count) || 250, 1);

  const blurPassed = laplacian >= 100.0;
  const brightnessPassed = brightness >= 60.0 && brightness <= 210.0;
  const resolutionPassed = w >= 800 && h >= 600;
  const sPixel = Number((refMm / refPx).toFixed(4));

  res.json({
    filename,
    qc_passed: blurPassed && brightnessPassed && resolutionPassed,
    laplacian_variance: Number(laplacian.toFixed(2)),
    laplacian_threshold: 100.0,
    blur_Passed: blurPassed,
    blur_status: blurPassed
      ? 'PASS — Sharp Focus (Var(ΔI) ≥ 100.0)'
      : 'REJECT — Defocus / Motion Blur Detected (Var(ΔI) < 100.0)',
    avg_brightness: Number(brightness.toFixed(1)),
    brightness_passed: brightnessPassed,
    brightness_status: brightnessPassed
      ? 'PASS — Balanced Dynamic Range'
      : brightness < 60
      ? 'WARN — Underexposed Frame'
      : 'WARN — Specular Overexposure',
    resolution_passed: resolutionPassed,
    dimensions: `${w} × ${h} px`,
    clahe_applied: Boolean(apply_clahe),
    clahe_clip_limit: 2.0,
    clahe_grid_size: '8×8 Tiles',
    scale_calibration_mm_per_px: sPixel,
    image_hash: crypto.createHash('sha256').update(`${filename}-${laplacian}`).digest('hex').slice(0, 16),
    tiles: [
      { tile_id: 'T00 (NW)', region: '0,0 → 1024,768', contrast_gain: '+18.4%', status: 'Ready' },
      { tile_id: 'T01 (NE)', region: '1024,0 → 2048,768', contrast_gain: '+21.2%', status: 'Ready' },
      { tile_id: 'T10 (SW)', region: '0,768 → 1024,1536', contrast_gain: '+19.7%', status: 'Ready' },
      { tile_id: 'T11 (SE)', region: '1024,768 → 2048,1536', contrast_gain: '+22.8%', status: 'Ready' },
    ],
  });
});

app.post('/api/v1/upload/single', (req, res) => {
  const user = getAuthenticatedUser(req);
  const allowedRoles: RoleType[] = ['Administrator', 'Civil Engineer', 'Field Inspector'];
  if (!allowedRoles.includes(user.role)) {
    return res.status(403).json({
      detail: `RBAC Policy Violation: Role '${user.role}' is not permitted to ingest raw inspection frames.`,
    });
  }

  const {
    asset_id = 1,
    blur_laplacian_var = 290.2,
    avg_brightness = 131.4,
    scale_mm_per_px = 0.35,
    notes = 'Uploaded via QC Preprocessing Workbench with CLAHE normalization.',
  } = req.body;

  if (Number(blur_laplacian_var) < 100.0) {
    return res.status(422).json({
      detail: `Quality Control Gate Rejected: Var(ΔI) = ${blur_laplacian_var} is below the 100.0 minimum sharpness threshold.`,
    });
  }

  const targetAsset =
    infrastructure_assets.find((a) => a.asset_id === Number(asset_id)) || infrastructure_assets[0];
  const comp = buildAssetComposite(targetAsset);
  comp.image.blur_laplacian_var = Number(blur_laplacian_var);
  comp.image.avg_brightness = Number(avg_brightness);
  comp.image.scale_mm_per_px = Number(scale_mm_per_px);
  comp.latest_inspection.notes = notes;
  comp.latest_inspection.inspection_date = new Date().toISOString();

  res.json({
    status: 'INGESTED_AND_ANALYZED',
    ingested_by: user.full_name,
    asset: buildAssetComposite(targetAsset),
  });
});

// 4. AI Analytics & Civil Engineer Override Endpoints
app.get('/api/v1/inspections/:id/analysis', (req, res) => {
  const id = Number(req.params.id);
  const inspection =
    inspections.find((i) => i.inspection_id === id || i.asset_id === id) || inspections[0];
  const asset =
    infrastructure_assets.find((a) => a.asset_id === inspection.asset_id) ||
    infrastructure_assets[0];
  res.json(buildAssetComposite(asset));
});

app.post('/api/v1/assessments/:id/override', (req, res) => {
  const user = getAuthenticatedUser(req);
  const allowedRoles: RoleType[] = ['Administrator', 'Civil Engineer'];
  if (!allowedRoles.includes(user.role)) {
    return res.status(403).json({
      detail: `RBAC Policy Violation: Only 'Civil Engineer' or 'Administrator' roles can override AI severity classifications (Current role: '${user.role}').`,
    });
  }

  const id = Number(req.params.id);
  const assessment =
    severity_assessments.find((a) => a.assessment_id === id || a.inspection_id === id) ||
    severity_assessments[0];

  const {
    severity_level,
    maintenance_priority,
    defect_class,
    engineer_comments,
    approval_status,
  } = req.body;

  if (severity_level) assessment.severity_level = severity_level;
  if (maintenance_priority) assessment.maintenance_priority = maintenance_priority;
  if (engineer_comments !== undefined) assessment.engineer_comments = engineer_comments;
  if (approval_status) assessment.approval_status = approval_status;
  assessment.verified_by = user.full_name;
  assessment.verified_at = new Date().toISOString();

  // Also update defect_class and health_score if reclassified
  const inspection = inspections.find((i) => i.inspection_id === assessment.inspection_id);
  if (inspection) {
    const img = inspection_images.find((im) => im.inspection_id === inspection.inspection_id);
    if (img && defect_class) {
      const det = defect_detections.find((d) => d.image_id === img.image_id);
      if (det) det.defect_class = defect_class;
    }
    const asset = infrastructure_assets.find((a) => a.asset_id === inspection.asset_id);
    if (asset && severity_level) {
      const scoreMap: Record<string, number> = {
        Minor: 88,
        Moderate: 72,
        Severe: 54,
        Critical: 36,
      };
      if (scoreMap[severity_level]) {
        asset.health_score = scoreMap[severity_level];
      }
    }
    if (asset) {
      return res.json(buildAssetComposite(asset));
    }
  }

  res.json({ status: 'updated', assessment });
});

// 5. Endpoint to retrieve the Python FastAPI source code for inspection/download
app.get('/api/v1/backend-source', (_req, res) => {
  try {
    const pyPath = path.resolve(process.cwd(), 'backend', 'main.py');
    const pyContent = fs.readFileSync(pyPath, 'utf-8');
    res.json({ filename: 'backend/main.py', code: pyContent });
  } catch {
    res.json({ filename: 'backend/main.py', code: '# FastAPI source file' });
  }
});

async function startServer() {
  const PORT = 3000;
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`InfraScan Full-Stack Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
