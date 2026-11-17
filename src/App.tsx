import React, { useState, useEffect, useRef, useMemo } from 'react';
import L from 'leaflet';
import {
  ShieldCheck,
  Upload,
  ScanLine,
  Search,
  Plus,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  ArrowUpRight,
  ArrowRight,
  Code2,
  Download,
  Printer,
  Copy,
  Check,
  Lock,
  ChevronRight,
  KeyRound,
  Building2,
  LogOut,
  FileCheck2,
  Unlock,
  UserPlus,
  Play,
  Layers,
  BarChart3,
  ClipboardCheck,
  SlidersHorizontal,
  GitCompare,
} from 'lucide-react';

// ============================================================================
// TYPES & DOMAIN INTERFACES
// ============================================================================

export type RoleType =
  | 'Administrator'
  | 'Infrastructure Manager'
  | 'Civil Engineer'
  | 'Field Inspector'
  | 'Maintenance Engineer';

export type DefectClassType =
  | 'D00 Longitudinal'
  | 'D10 Transverse'
  | 'D20 Alligator'
  | 'D40 Pothole'
  | 'Spalling';

export type SeverityLevelType = 'Minor' | 'Moderate' | 'Severe' | 'Critical';
export type PriorityType = 'Routine' | 'Scheduled' | 'Urgent' | 'Immediate';

export interface AuthenticatedUser {
  user_id: number;
  email: string;
  full_name: string;
  role: RoleType;
  token: string;
}

export interface GradCamHotspot {
  cx: number;
  cy: number;
  rx: number;
  ry: number;
  weight: number;
  label: string;
}

export interface AssetComposite {
  asset_id: number;
  asset_name: string;
  asset_type: 'Bridge' | 'Pavement' | 'Vertical Building' | 'Tunnel' | 'Hydraulic Dam';
  material_type: string;
  construction_year: number;
  latitude: number;
  longitude: number;
  responsible_department: string;
  health_score: number;
  inspector_name: string;
  latest_inspection: {
    inspection_id: number;
    asset_id: number;
    inspector_id: number;
    inspection_date: string;
    status: string;
    notes: string;
  };
  image: {
    image_id: number;
    inspection_id: number;
    storage_url: string;
    width_px: number;
    height_px: number;
    blur_laplacian_var: number;
    avg_brightness: number;
    image_hash: string;
    scale_mm_per_px: number;
    is_calibrated?: boolean;
  };
  detection: {
    detection_id: number;
    image_id: number;
    defect_class: DefectClassType;
    confidence_score: number;
    bbox_xmin: number;
    bbox_ymin: number;
    bbox_xmax: number;
    bbox_ymax: number;
    baseline_classification?: {
      label: 'Cracked' | 'Uncracked';
      confidence: number;
      model: string;
      dataset: string;
    };
    multi_class_boxes?: Array<{
      label: DefectClassType;
      confidence: number;
      xmin: number;
      ymin: number;
      xmax: number;
      ymax: number;
    }>;
  };
  segmentation: {
    segmentation_id: number;
    detection_id: number;
    crack_length_mm: number;
    avg_width_mm: number;
    max_width_mm: number;
    crack_density_pct: number;
    svg_mask_paths: string[];
    gradcam_hotspots: GradCamHotspot[];
    segmentation_metrics?: {
      iou: number;
      dice: number;
      pixel_accuracy: number;
      f1_score: number;
      boundary_f1: number;
      mask_area_px: number;
    };
    crack_geometry?: {
      tortuosity_index: number;
      dominant_orientation_deg: number;
      branch_count: number;
      junction_nodes: number;
      skeleton_length_px: number;
      mean_aperture_px: number;
      max_aperture_px: number;
    };
  };
  assessment: {
    assessment_id: number;
    inspection_id: number;
    severity_level: SeverityLevelType;
    maintenance_priority: PriorityType;
    engineer_comments: string;
    verified_by?: string;
    verified_at?: string;
    approval_status: 'AI Baseline' | 'Approved' | 'Reclassified' | 'Escalated';
  };
  maintenance_workflow?: {
    asset_id: number;
    work_order_id: string;
    workflow_stage: 'Triaged' | 'Scheduled' | 'Crew Dispatched' | 'In Remediation' | 'Verified Closed';
    assigned_team: string;
    sla_days: number;
    estimated_cost_usd: number;
    updated_at: string;
  };
}

export interface HistoryEntry {
  inspection_id: number;
  inspection_date: string;
  inspector: string;
  defect_class: string;
  max_width_mm: number;
  crack_density_pct: number;
  severity_level: SeverityLevelType;
  status: string;
  notes: string;
}

export interface QcValidationResult {
  filename: string;
  qc_passed: boolean;
  laplacian_variance: number;
  laplacian_threshold: number;
  blur_status: string;
  avg_brightness: number;
  brightness_status: string;
  resolution_passed: boolean;
  dimensions: string;
  clahe_applied: boolean;
  clahe_clip_limit: number;
  clahe_grid_size: string;
  is_calibrated?: boolean;
  calibration_status?: string;
  scale_calibration_mm_per_px: number;
  image_hash: string;
  tiles: Array<{
    tile_id: string;
    region: string;
    contrast_gain: string;
    status: string;
  }>;
}

export interface ModelBenchmarkEntry {
  model_id: string;
  name: string;
  family: 'Baseline Classification' | 'Object Detection' | 'Segmentation & Advanced CNN/Transformer';
  architecture: 'CNN' | 'YOLO' | 'U-Net' | 'Transformer' | 'Hybrid';
  dataset: string;
  primary_metric_name: string;
  primary_metric_value: number;
  secondary_metric_name: string;
  secondary_metric_value: number;
  precision: number;
  recall: number;
  f1_or_dice: number;
  latency_ms: number;
  params_m: number;
  epochs_trained: number;
  is_selected: boolean;
  selection_note: string;
  last_evaluated_at: string;
}

export interface BenchmarksPayload {
  models: ModelBenchmarkEntry[];
  epoch_curve: Array<{
    epoch: number;
    unet_iou: number;
    segformer_iou: number;
    yolo_map: number;
    resnet_acc: number;
  }>;
  per_class_ap: Array<{
    defect_class: string;
    ap_yolov11: number;
    iou_segformer: number;
    iou_unet: number;
    samples: number;
  }>;
  summary: {
    best_baseline_classifier: string;
    best_detector: string;
    best_baseline_segmentation: string;
    best_advanced_transformer: string;
  };
}

export interface AnalyticsSummary {
  total_assets: number;
  calibrated_assets: number;
  total_crack_length_mm: number;
  max_aperture_mm: number;
  mean_crack_density_pct: number;
  total_remediation_budget_usd: number;
  severity_breakdown: Array<{ level: string; count: number; pct: number }>;
  defect_class_breakdown: Array<{
    defect_class: string;
    count: number;
    avg_max_width_mm: number;
    avg_density_pct: number;
  }>;
  asset_type_risk: Array<{
    asset_type: string;
    count: number;
    avg_health: number;
    total_crack_length_mm: number;
  }>;
}

export interface TasksStatusPayload {
  tasks_completed: number;
  tasks_total: number;
  outcomes_operational: number;
  outcomes_total: number;
  tasks: Array<{
    id: string;
    title: string;
    category: 'Image & CV' | 'Models & XAI' | 'GIS & Maintenance';
    workspace: WorkspaceTab;
    endpoint: string;
    evidence: string;
    status: string;
  }>;
  outcomes: Array<{
    id: string;
    title: string;
    linked_tasks: string[];
    status: string;
  }>;
}

type WorkspaceTab = 'assets' | 'qc' | 'inspection' | 'map' | 'reports';
type ViewportLayer = 'raw' | 'yolo' | 'unet' | 'gradcam';

// Pure Role Permissions Matrix (No hardcoded example user names)
const ROLE_PERMISSIONS_MAP: Record<
  RoleType,
  {
    requiresAdminKey: boolean;
    canRegisterAsset: boolean;
    canIngestImage: boolean;
    canOverrideAI: boolean;
    summary: string;
  }
> = {
  'Civil Engineer': {
    requiresAdminKey: false,
    canRegisterAsset: true,
    canIngestImage: true,
    canOverrideAI: true,
    summary: 'PE Sign-Off Authority · AI Severity Override · Defect Reclassification',
  },
  'Infrastructure Manager': {
    requiresAdminKey: false,
    canRegisterAsset: true,
    canIngestImage: false,
    canOverrideAI: false,
    summary: 'Portfolio Asset Registration · Geospatial Budget & Priority Oversight',
  },
  'Field Inspector': {
    requiresAdminKey: false,
    canRegisterAsset: false,
    canIngestImage: true,
    canOverrideAI: false,
    summary: 'Optical Frame Upload · Laplacian Blur & CLAHE Preprocessing Gating',
  },
  'Maintenance Engineer': {
    requiresAdminKey: false,
    canRegisterAsset: false,
    canIngestImage: false,
    canOverrideAI: false,
    summary: 'Read-Only Remediation Queue · Crack Geometry & Field Dispatch Export',
  },
  Administrator: {
    requiresAdminKey: true,
    canRegisterAsset: true,
    canIngestImage: true,
    canOverrideAI: true,
    summary: 'Full System & RBAC Governance · Asset Registry · AI Override Authority',
  },
};

// High-Contrast Light Mode Semantic Helper (Zero Pill Enclosures)
function getSeverityMeta(severity: SeverityLevelType) {
  switch (severity) {
    case 'Critical':
      return {
        textClass: 'text-red-700',
        hex: '#DC2626',
        icon: XCircle,
        label: 'Critical Severity',
      };
    case 'Severe':
      return {
        textClass: 'text-amber-700',
        hex: '#D97706',
        icon: AlertTriangle,
        label: 'Severe Damage',
      };
    case 'Moderate':
      return {
        textClass: 'text-yellow-700',
        hex: '#CA8A04',
        icon: AlertTriangle,
        label: 'Moderate Defect',
      };
    case 'Minor':
    default:
      return {
        textClass: 'text-emerald-700',
        hex: '#16A34A',
        icon: CheckCircle2,
        label: 'Minor Hairline',
      };
  }
}

// Resilient Image Component with Zero-Broken-Image Policy
const ResilientInfrastructureImage: React.FC<{
  src: string;
  alt: string;
  className?: string;
  style?: React.CSSProperties;
}> = ({ src, alt, className = '', style }) => {
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    setHasError(false);
  }, [src]);

  if (hasError || !src) {
    return (
      <div
        className={`flex flex-col items-center justify-center bg-slate-100 text-slate-600 p-6 ${className}`}
        style={style}
      >
        <ScanLine className="w-8 h-8 text-blue-600 mb-2 stroke-[1.5]" />
        <span className="text-xs font-medium text-slate-800 text-center">{alt}</span>
        <span className="text-[11px] font-mono text-slate-500 mt-1">
          Orthogonal Photogrammetry Surface Tile
        </span>
      </div>
    );
  }

  return (
    <img
      src={src}
      alt={alt}
      referrerPolicy="no-referrer"
      onError={() => setHasError(true)}
      className={className}
      style={style}
    />
  );
};

// Enterprise Geometric Structural Emblem Logo
const EnterpriseBrandLogo: React.FC<{ size?: 'sm' | 'lg' }> = ({ size = 'sm' }) => {
  const boxClass = size === 'lg' ? 'w-10 h-10' : 'w-7 h-7';
  return (
    <div
      className={`${boxClass} rounded-lg bg-slate-900 text-white flex items-center justify-center shrink-0 shadow-xs`}
      aria-hidden="true"
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        className={size === 'lg' ? 'w-5 h-5' : 'w-4 h-4'}
        stroke="currentColor"
        strokeWidth="2"
      >
        <path d="M3 20h18" strokeLinecap="round" />
        <path d="M6 20V9l6-5 6 5v11" strokeLinejoin="round" />
        <path d="M9 14l2.5-2.5L14 14l2.5-3" stroke="#38BDF8" strokeLinecap="round" />
      </svg>
    </div>
  );
};

// ============================================================================
// MAIN APPLICATION
// ============================================================================

export default function App() {
  // Authentication & Login / Create Account Screen State
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const [authFullName, setAuthFullName] = useState<string>('');
  const [authEmail, setAuthEmail] = useState<string>('');
  const [authPassword, setAuthPassword] = useState<string>('');
  const [authSelectedRole, setAuthSelectedRole] = useState<RoleType>('Civil Engineer');
  const [authAdminAccessKey, setAuthAdminAccessKey] = useState<string>('');
  const [isAuthenticating, setIsAuthenticating] = useState<boolean>(false);
  const [authError, setAuthError] = useState<string | null>(null);

  // Admin Role Override Modal State (Active Role is locked after sign-in unless Admin overrides)
  const [pendingOverrideRole, setPendingOverrideRole] = useState<RoleType | null>(null);
  const [adminOverrideKeyInput, setAdminOverrideKeyInput] = useState<string>('');
  const [adminOverrideError, setAdminOverrideError] = useState<string | null>(null);
  const [isOverridingRole, setIsOverridingRole] = useState<boolean>(false);

  // Navigation & Active Asset State
  const [activeWorkspace, setActiveWorkspace] = useState<WorkspaceTab>('assets');
  const [assets, setAssets] = useState<AssetComposite[]>([]);
  const [selectedAssetId, setSelectedAssetId] = useState<number>(1);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [toastMessage, setToastMessage] = useState<{ text: string; type: 'info' | 'error' } | null>(
    null
  );

  // Active User Session & RBAC State
  const [currentUser, setCurrentUser] = useState<AuthenticatedUser>({
    user_id: 1,
    email: '',
    full_name: 'Authenticated Operator',
    role: 'Civil Engineer',
    token: '',
  });

  // Workspace 1: Asset Hub Filters & Modals
  const [assetLayout, setAssetLayout] = useState<'table' | 'grid' | 'priority'>('table');
  const [filterType, setFilterType] = useState<string>('All');
  const [filterSeverity, setFilterSeverity] = useState<string>('All');
  const [filterPriority, setFilterPriority] = useState<string>('All');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [showNewAssetModal, setShowNewAssetModal] = useState<boolean>(false);
  const [historyDrawerAsset, setHistoryDrawerAsset] = useState<{
    asset: AssetComposite;
    history: HistoryEntry[];
  } | null>(null);
  const [compareIdxA, setCompareIdxA] = useState<number>(2);
  const [compareIdxB, setCompareIdxB] = useState<number>(0);

  // New Asset Form State
  const [newAssetForm, setNewAssetForm] = useState({
    asset_name: '',
    asset_type: 'Bridge',
    material_type: 'Reinforced Concrete (C40/50)',
    construction_year: 2015,
    latitude: 37.8044,
    longitude: -122.2712,
    responsible_department: 'Structural Engineering Division',
  });

  // Workspace 2: Image QC & Preprocessing Workbench State
  const [qcSelectedPresetId, setQcSelectedPresetId] = useState<number>(1);
  const [qcSimulatedBlurMode, setQcSimulatedBlurMode] = useState<'sharp' | 'motion_blur'>('sharp');
  const [qcCustomImageSrc, setQcCustomImageSrc] = useState<string | null>(null);
  const [qcCustomFilename, setQcCustomFilename] = useState<string>('span_b14_nadir_tile.jpg');
  const [qcClaheEnabled, setQcClaheEnabled] = useState<boolean>(true);
  const [qcIsCalibrated, setQcIsCalibrated] = useState<boolean>(true);
  const [qcRefMm, setQcRefMm] = useState<number>(100);
  const [qcRefPx, setQcRefPx] = useState<number>(294);
  const [qcResult, setQcResult] = useState<QcValidationResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Workspace 3: AI Inspection & Quantitative Analysis State
  const [inspectionSubTab, setInspectionSubTab] = useState<'viewport' | 'benchmarks'>('viewport');
  const [activeDetectorModel, setActiveDetectorModel] = useState<'YOLOv11x' | 'YOLOv8x' | 'RT-DETR-L'>('YOLOv11x');
  const [activeSegModel, setActiveSegModel] = useState<'SegFormer-B2' | 'Attention U-Net' | 'Swin-UNet'>('SegFormer-B2');
  const [viewportLayer, setViewportLayer] = useState<ViewportLayer>('yolo');
  const [overlayOpacity, setOverlayOpacity] = useState<number>(88);
  const [overrideSeverity, setOverrideSeverity] = useState<SeverityLevelType>('Severe');
  const [overridePriority, setOverridePriority] = useState<PriorityType>('Urgent');
  const [overrideDefectClass, setOverrideDefectClass] =
    useState<DefectClassType>('D20 Alligator');
  const [overrideComments, setOverrideComments] = useState<string>('');
  const [isSavingOverride, setIsSavingOverride] = useState<boolean>(false);

  // Model Benchmarks, Training & Evaluation State
  const [benchmarksData, setBenchmarksData] = useState<BenchmarksPayload | null>(null);
  const [trainModelId, setTrainModelId] = useState<string>('segformer-b2');
  const [trainEpochs, setTrainEpochs] = useState<number>(5);
  const [isTrainingModel, setIsTrainingModel] = useState<boolean>(false);

  // Workspace 4: Geospatial Damage Mapping State
  const [mapSeverityFilter, setMapSeverityFilter] = useState<string>('All');
  const [showHeatmapLayer, setShowHeatmapLayer] = useState<boolean>(true);
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const leafletMapRef = useRef<L.Map | null>(null);
  const markersLayerRef = useRef<L.LayerGroup | null>(null);

  // Workspace 5: Analytics, Tasks Matrix & Compliance Report Export
  const [reportsSubTab, setReportsSubTab] = useState<'analytics' | 'compliance'>('analytics');
  const [analyticsSummary, setAnalyticsSummary] = useState<AnalyticsSummary | null>(null);
  const [tasksStatus, setTasksStatus] = useState<TasksStatusPayload | null>(null);
  const [taskCategoryFilter, setTaskCategoryFilter] = useState<string>('All');
  const [taskSearchQuery, setTaskSearchQuery] = useState<string>('');
  const [reportScope, setReportScope] = useState<'all' | 'critical_severe'>('all');
  const [includeXaiAppendix, setIncludeXaiAppendix] = useState<boolean>(true);
  const [includePostgisCoords, setIncludePostgisCoords] = useState<boolean>(true);
  const [engineerSignOffName, setEngineerSignOffName] = useState<string>('');
  const [engineerLicenseNo, setEngineerLicenseNo] = useState<string>('PE-C84920');
  const [fastApiSource, setFastApiSource] = useState<string>('');
  const [showBackendDrawer, setShowBackendDrawer] = useState<boolean>(false);
  const [copiedCode, setCopiedCode] = useState<boolean>(false);

  const showNotice = (text: string, type: 'info' | 'error' = 'info') => {
    setToastMessage({ text, type });
    setTimeout(() => {
      setToastMessage((prev) => (prev?.text === text ? null : prev));
    }, 4500);
  };

  // Initial Data Fetch (/api/v1/assets, /api/v1/backend-source, /api/v1/models/benchmarks, /api/v1/analytics/summary, /api/v1/tasks/status)
  useEffect(() => {
    async function bootstrap() {
      setIsLoading(true);
      try {
        const [assetsRes, srcRes, benchRes, analyticsRes, tasksRes] = await Promise.all([
          fetch('/api/v1/assets'),
          fetch('/api/v1/backend-source'),
          fetch('/api/v1/models/benchmarks'),
          fetch('/api/v1/analytics/summary'),
          fetch('/api/v1/tasks/status'),
        ]);
        if (assetsRes.ok) {
          const assetsData: AssetComposite[] = await assetsRes.json();
          setAssets(assetsData);
          if (assetsData.length > 0) {
            setSelectedAssetId(assetsData[0].asset_id);
          }
        }
        if (srcRes.ok) {
          const srcData = await srcRes.json();
          setFastApiSource(srcData.code || '');
        }
        if (benchRes.ok) {
          setBenchmarksData(await benchRes.json());
        }
        if (analyticsRes.ok) {
          setAnalyticsSummary(await analyticsRes.json());
        }
        if (tasksRes.ok) {
          setTasksStatus(await tasksRes.json());
        }
      } catch (err) {
        console.error('Bootstrap error:', err);
      } finally {
        setIsLoading(false);
      }
    }
    bootstrap();
  }, []);

  const refreshAnalytics = async () => {
    try {
      const res = await fetch('/api/v1/analytics/summary');
      if (res.ok) setAnalyticsSummary(await res.json());
    } catch {
      // silent
    }
  };

  // Selected Asset Composite
  const selectedAsset = useMemo(() => {
    return assets.find((a) => a.asset_id === selectedAssetId) || assets[0] || null;
  }, [assets, selectedAssetId]);

  // Sync Override Controls when selectedAsset changes
  useEffect(() => {
    if (selectedAsset) {
      setOverrideSeverity(selectedAsset.assessment.severity_level);
      setOverridePriority(selectedAsset.assessment.maintenance_priority);
      setOverrideDefectClass(selectedAsset.detection.defect_class);
      setOverrideComments(selectedAsset.assessment.engineer_comments);
    }
  }, [selectedAsset]);

  // Trigger QC Validation whenever QC inputs change
  useEffect(() => {
    const presetAsset = assets.find((a) => a.asset_id === qcSelectedPresetId) || assets[0];
    if (!presetAsset) return;

    const baseLaplacian =
      qcSimulatedBlurMode === 'motion_blur' ? 42.6 : presetAsset.image.blur_laplacian_var;
    const baseBrightness =
      qcSimulatedBlurMode === 'motion_blur' ? 88.4 : presetAsset.image.avg_brightness;

    async function runValidation() {
      try {
        const res = await fetch('/api/v1/preprocess/validate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${currentUser.token}`,
          },
          body: JSON.stringify({
            filename: qcCustomImageSrc
              ? qcCustomFilename
              : `${presetAsset.asset_name.toLowerCase().replace(/\s+/g, '_')}.jpg`,
            width_px: presetAsset.image.width_px,
            height_px: presetAsset.image.height_px,
            blur_laplacian_var: baseLaplacian,
            avg_brightness: baseBrightness,
            reference_distance_mm: qcRefMm,
            reference_pixel_count: qcRefPx,
            apply_clahe: qcClaheEnabled,
            is_calibrated: qcIsCalibrated,
          }),
        });
        if (res.ok) {
          const data: QcValidationResult = await res.json();
          setQcResult(data);
        }
      } catch (err) {
        console.error(err);
      }
    }
    runValidation();
  }, [
    qcSelectedPresetId,
    qcSimulatedBlurMode,
    qcClaheEnabled,
    qcIsCalibrated,
    qcRefMm,
    qcRefPx,
    qcCustomImageSrc,
    qcCustomFilename,
    assets,
    currentUser.token,
  ]);

  // Handle Log In or Create Account Submission
  const handleAuthSubmit = async (e?: React.FormEvent, isSso = false) => {
    if (e) e.preventDefault();
    setAuthError(null);

    if (!authEmail.trim() || !authPassword.trim()) {
      setAuthError('Please enter your email address and password.');
      return;
    }

    if (authMode === 'register' && !authFullName.trim()) {
      setAuthError('Please enter your full name to create an account.');
      return;
    }

    setIsAuthenticating(true);

    try {
      const endpoint =
        authMode === 'register' ? '/api/v1/auth/register' : '/api/v1/auth/login';
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          full_name: authFullName.trim(),
          email: authEmail.trim(),
          username: authEmail.trim(),
          password: authPassword,
          role: authSelectedRole,
          admin_access_key: authAdminAccessKey.trim(),
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        setAuthError(data.detail || 'Authentication failed.');
        return;
      }

      setCurrentUser({
        user_id: data.user.user_id,
        email: data.user.email,
        full_name: data.user.full_name,
        role: data.user.role,
        token: data.access_token,
      });
      setEngineerSignOffName(data.user.full_name);
      setIsAuthenticated(true);
      setAuthAdminAccessKey('');
      showNotice(
        isSso
          ? `Authenticated via Enterprise SAML 2.0 SSO as ${data.user.full_name}. Active role locked to ${data.user.role}.`
          : authMode === 'register'
          ? `Account created for ${data.user.full_name}. Active role locked to ${data.user.role}.`
          : `Signed in as ${data.user.full_name}. Active role locked to ${data.user.role}.`
      );
    } catch {
      setAuthError('Unable to reach authentication service.');
    } finally {
      setIsAuthenticating(false);
    }
  };

  // Attempt to change active role after sign-in -> Requires Admin Override (Access Key 1234)
  const handleRoleDropdownSelectInWorkspace = (requestedRole: RoleType) => {
    if (requestedRole === currentUser.role) return;
    setPendingOverrideRole(requestedRole);
    setAdminOverrideKeyInput('');
    setAdminOverrideError(null);
  };

  // Confirm Admin Role Override (POST /api/v1/auth/override-role with Key 1234)
  const handleConfirmAdminRoleOverride = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!pendingOverrideRole) return;
    setIsOverridingRole(true);
    setAdminOverrideError(null);

    try {
      const res = await fetch('/api/v1/auth/override-role', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${currentUser.token}`,
        },
        body: JSON.stringify({
          target_role: pendingOverrideRole,
          admin_access_key: adminOverrideKeyInput.trim(),
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        setAdminOverrideError(
          data.detail || 'Admin Override Denied: Invalid Administrator Access Key.'
        );
        return;
      }

      setCurrentUser({
        user_id: data.user.user_id,
        email: data.user.email,
        full_name: data.user.full_name,
        role: data.user.role,
        token: data.access_token,
      });
      setPendingOverrideRole(null);
      setAdminOverrideKeyInput('');
      showNotice(
        `Administrator Override Approved: Active session role switched to ${data.user.role}.`
      );
    } catch {
      setAdminOverrideError('Network error verifying Administrator Override.');
    } finally {
      setIsOverridingRole(false);
    }
  };

  // Filtered Assets for Workspace 1
  const filteredAssets = useMemo(() => {
    return assets.filter((item) => {
      const matchesType = filterType === 'All' || item.asset_type === filterType;
      const matchesSeverity =
        filterSeverity === 'All' || item.assessment.severity_level === filterSeverity;
      const matchesPriority =
        filterPriority === 'All' || item.assessment.maintenance_priority === filterPriority;
      const matchesSearch =
        !searchQuery.trim() ||
        item.asset_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.responsible_department.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.detection.defect_class.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesType && matchesSeverity && matchesPriority && matchesSearch;
    });
  }, [assets, filterType, filterSeverity, filterPriority, searchQuery]);

  // Open Asset History Drawer
  const handleOpenHistory = async (assetId: number) => {
    try {
      const res = await fetch(`/api/v1/assets/${assetId}/history`, {
        headers: { Authorization: `Bearer ${currentUser.token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setHistoryDrawerAsset(data);
      }
    } catch {
      showNotice('Unable to load asset inspection history', 'error');
    }
  };

  // Register New Infrastructure Asset (POST /api/v1/assets)
  const handleCreateAsset = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await fetch('/api/v1/assets', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${currentUser.token}`,
        },
        body: JSON.stringify(newAssetForm),
      });
      if (!res.ok) {
        const err = await res.json();
        showNotice(err.detail || 'RBAC Authorization failed', 'error');
        return;
      }
      const created: AssetComposite = await res.json();
      setAssets((prev) => [...prev, created]);
      setSelectedAssetId(created.asset_id);
      setShowNewAssetModal(false);
      setNewAssetForm({
        asset_name: '',
        asset_type: 'Bridge',
        material_type: 'Reinforced Concrete (C40/50)',
        construction_year: 2015,
        latitude: 37.8044,
        longitude: -122.2712,
        responsible_department: 'Structural Engineering Division',
      });
      showNotice(`Registered "${created.asset_name}" into PostGIS asset registry.`);
    } catch {
      showNotice('Network error while registering asset', 'error');
    }
  };

  // Handle Custom Image Upload in QC Workbench
  const handleFileUpload = (file: File) => {
    const reader = new FileReader();
    reader.onload = (ev) => {
      const dataUrl = ev.target?.result as string;
      setQcCustomImageSrc(dataUrl);
      setQcCustomFilename(file.name);
      setQcSimulatedBlurMode('sharp');
      showNotice(`Loaded local inspection frame "${file.name}" for Laplacian QC analysis.`);
    };
    reader.readAsDataURL(file);
  };

  // Ingest Frame to AI Pipeline (POST /api/v1/upload/single)
  const handleIngestFrameToPipeline = async () => {
    if (!qcResult) return;
    try {
      const res = await fetch('/api/v1/upload/single', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${currentUser.token}`,
        },
        body: JSON.stringify({
          asset_id: qcSelectedPresetId,
          blur_laplacian_var: qcResult.laplacian_variance,
          avg_brightness: qcResult.avg_brightness,
          scale_mm_per_px: qcResult.scale_calibration_mm_per_px,
          is_calibrated: qcIsCalibrated,
          notes: `QC Validated (Var(ΔI)=${qcResult.laplacian_variance}, ${
            qcIsCalibrated
              ? `Scale=${qcResult.scale_calibration_mm_per_px} mm/px`
              : 'Uncalibrated px-only mode'
          }, CLAHE=${qcResult.clahe_applied ? 'ON' : 'OFF'})`,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        showNotice(data.detail || 'QC Gate or RBAC policy rejected ingestion', 'error');
        return;
      }
      setAssets((prev) =>
        prev.map((a) => (a.asset_id === data.asset.asset_id ? data.asset : a))
      );
      setSelectedAssetId(data.asset.asset_id);
      setActiveWorkspace('inspection');
      setInspectionSubTab('viewport');
      refreshAnalytics();
      showNotice(
        `Frame passed QC Gating and routed to YOLOv11 + U-Net inference for ${data.asset.asset_name}.`
      );
    } catch {
      showNotice('Error submitting frame to ingestion endpoint', 'error');
    }
  };

  // Train / Evaluate Model Handler (POST /api/v1/models/train)
  const handleTrainModel = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsTrainingModel(true);
    try {
      const res = await fetch('/api/v1/models/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${currentUser.token}`,
        },
        body: JSON.stringify({ model_id: trainModelId, additional_epochs: trainEpochs }),
      });
      if (res.ok) {
        const data = await res.json();
        setBenchmarksData((prev) => (prev ? { ...prev, models: data.models } : prev));
        showNotice(
          `Completed ${trainEpochs}-epoch evaluation pass for ${data.updated_model.name} (${data.updated_model.primary_metric_name}: ${data.updated_model.primary_metric_value}%).`
        );
      }
    } finally {
      setIsTrainingModel(false);
    }
  };

  // Select Active Production Model (POST /api/v1/models/select)
  const handleSelectModel = async (modelId: string) => {
    try {
      const res = await fetch('/api/v1/models/select', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${currentUser.token}`,
        },
        body: JSON.stringify({ model_id: modelId }),
      });
      if (res.ok) {
        const data = await res.json();
        setBenchmarksData((prev) => (prev ? { ...prev, models: data.models } : prev));
        showNotice(`Selected "${data.selected_model.name}" as active benchmark model.`);
      }
    } catch {
      showNotice('Unable to select model', 'error');
    }
  };

  // Update Maintenance Prioritization Workflow Stage (POST /api/v1/maintenance/:assetId/workflow)
  const handleUpdateWorkflowStage = async (
    assetId: number,
    workflowStage: 'Triaged' | 'Scheduled' | 'Crew Dispatched' | 'In Remediation' | 'Verified Closed'
  ) => {
    try {
      const res = await fetch(`/api/v1/maintenance/${assetId}/workflow`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${currentUser.token}`,
        },
        body: JSON.stringify({ workflow_stage: workflowStage }),
      });
      const data = await res.json();
      if (!res.ok) {
        showNotice(data.detail || 'RBAC restriction updating work order', 'error');
        return;
      }
      setAssets((prev) => prev.map((a) => (a.asset_id === data.asset_id ? data : a)));
      refreshAnalytics();
      showNotice(
        `Work Order ${data.maintenance_workflow?.work_order_id} moved to "${workflowStage}".`
      );
    } catch {
      showNotice('Network error updating maintenance workflow', 'error');
    }
  };

  // Export Generated Binary Crack Mask SVG
  const handleDownloadMaskSvg = (asset: AssetComposite) => {
    const paths = asset.segmentation.svg_mask_paths
      .map(
        (d) =>
          `  <path d="${d}" fill="none" stroke="#FFFFFF" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" />`
      )
      .join('\n');
    const svgContent = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="${asset.image.width_px}" height="${asset.image.height_px}">\n  <rect width="100" height="100" fill="#000000" />\n${paths}\n</svg>`;
    const blob = new Blob([svgContent], { type: 'image/svg+xml;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute(
      'download',
      `crack_mask_asset_${asset.asset_id}_${asset.detection.defect_class.replace(/\s+/g, '_')}.svg`
    );
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    showNotice(`Generated and downloaded binary crack mask for ${asset.asset_name}.`);
  };

  // Export Model Evaluation Report JSON
  const handleExportModelReport = () => {
    if (!benchmarksData) return;
    const report = {
      generated_at: new Date().toISOString(),
      evaluator: currentUser.full_name,
      summary: benchmarksData.summary,
      models: benchmarksData.models,
      per_class_ap: benchmarksData.per_class_ap,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `infrascan_model_evaluation_report_${Date.now()}.json`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    showNotice('Exported Model Evaluation & Benchmark Report (JSON).');
  };

  // Civil Engineer Override Handler (POST /api/v1/assessments/{id}/override)
  const handleEngineeringOverride = async (
    statusType: 'Approved' | 'Reclassified' | 'Escalated'
  ) => {
    if (!selectedAsset) return;
    setIsSavingOverride(true);
    try {
      const res = await fetch(
        `/api/v1/assessments/${selectedAsset.assessment.assessment_id}/override`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${currentUser.token}`,
          },
          body: JSON.stringify({
            severity_level: overrideSeverity,
            maintenance_priority: overridePriority,
            defect_class: overrideDefectClass,
            engineer_comments: overrideComments,
            approval_status: statusType,
          }),
        }
      );
      const data = await res.json();
      if (!res.ok) {
        showNotice(data.detail || 'RBAC Override Denied', 'error');
        return;
      }
      setAssets((prev) => prev.map((a) => (a.asset_id === data.asset_id ? data : a)));
      refreshAnalytics();
      showNotice(
        `Assessment #${selectedAsset.assessment.assessment_id} marked "${statusType}" by ${currentUser.full_name}.`
      );
    } finally {
      setIsSavingOverride(false);
    }
  };

  // Initialize & Sync Light-Mode Leaflet Map when Workspace 4 ('map') is active
  useEffect(() => {
    if (!isAuthenticated || activeWorkspace !== 'map' || !mapContainerRef.current) return;

    if (!leafletMapRef.current) {
      const map = L.map(mapContainerRef.current, {
        center: [37.815, -122.32],
        zoom: 10,
        zoomControl: true,
      });

      L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap contributors &copy; CARTO PostGIS Spatial Layer',
        maxZoom: 19,
      }).addTo(map);

      markersLayerRef.current = L.layerGroup().addTo(map);
      leafletMapRef.current = map;
    }

    const map = leafletMapRef.current;
    const layerGroup = markersLayerRef.current;
    if (!map || !layerGroup) return;

    setTimeout(() => {
      map.invalidateSize();
    }, 100);

    layerGroup.clearLayers();

    const visibleMapAssets = assets.filter(
      (a) => mapSeverityFilter === 'All' || a.assessment.severity_level === mapSeverityFilter
    );

    visibleMapAssets.forEach((asset) => {
      const sev = getSeverityMeta(asset.assessment.severity_level);

      if (showHeatmapLayer) {
        const radiusMeters = Math.max(asset.segmentation.crack_density_pct * 160, 800);
        L.circle([asset.latitude, asset.longitude], {
          radius: radiusMeters,
          color: sev.hex,
          weight: 1.5,
          fillColor: sev.hex,
          fillOpacity: 0.16,
        }).addTo(layerGroup);
      }

      const customIcon = L.divIcon({
        className: 'custom-infrascan-pin-light',
        html: `
          <div style="
            display: flex;
            align-items: center;
            gap: 6px;
            background: #FFFFFF;
            border: 1.5px solid ${sev.hex};
            color: #0F172A;
            padding: 3px 8px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 600;
            white-space: nowrap;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.14);
          ">
            <span style="width: 8px; height: 8px; border-radius: 2px; background: ${sev.hex}; display: inline-block;"></span>
            <span>${asset.assessment.severity_level.toUpperCase()} · ${asset.segmentation.crack_density_pct}%</span>
          </div>
        `,
        iconSize: [120, 28],
        iconAnchor: [60, 14],
      });

      const marker = L.marker([asset.latitude, asset.longitude], { icon: customIcon });
      marker.on('click', () => {
        setSelectedAssetId(asset.asset_id);
      });

      marker.bindPopup(`
        <div style="min-width: 220px; padding: 2px;">
          <div style="font-size: 11px; color: #64748B; margin-bottom: 2px;">
            ${asset.asset_type} · Built ${asset.construction_year}
          </div>
          <div style="font-size: 14px; font-weight: 600; color: #0F172A; margin-bottom: 6px;">
            ${asset.asset_name}
          </div>
          <div style="font-size: 12px; font-family: monospace; color: ${sev.hex}; font-weight: 600; margin-bottom: 6px;">
            ${asset.detection.defect_class} · ${asset.assessment.severity_level} (${asset.assessment.maintenance_priority})
          </div>
          <div style="font-size: 11px; color: #475569; font-family: monospace;">
            Length: ${asset.segmentation.crack_length_mm} mm · Max Width: ${asset.segmentation.max_width_mm} mm
          </div>
        </div>
      `);

      marker.addTo(layerGroup);
    });
  }, [isAuthenticated, activeWorkspace, assets, mapSeverityFilter, showHeatmapLayer]);

  // Clean up Leaflet instance when leaving map view or logging out
  useEffect(() => {
    if (!isAuthenticated || activeWorkspace !== 'map') {
      if (leafletMapRef.current) {
        leafletMapRef.current.remove();
        leafletMapRef.current = null;
        markersLayerRef.current = null;
      }
    }
  }, [isAuthenticated, activeWorkspace]);

  // CSV Download Handler
  const handleDownloadCsv = () => {
    const rowsForExport =
      reportScope === 'critical_severe'
        ? assets.filter(
            (a) =>
              a.assessment.severity_level === 'Critical' ||
              a.assessment.severity_level === 'Severe'
          )
        : assets;

    const headers = [
      'Asset_ID',
      'Asset_Name',
      'Asset_Type',
      'Material',
      'Latitude',
      'Longitude',
      'Defect_Class',
      'YOLO_Confidence',
      'Crack_Length_mm',
      'Avg_Width_mm',
      'Max_Width_mm',
      'Crack_Density_Pct',
      'Severity_Level',
      'Maintenance_Priority',
      'Approval_Status',
      'Verified_By',
    ];

    const csvLines = [
      headers.join(','),
      ...rowsForExport.map((a) =>
        [
          a.asset_id,
          `"${a.asset_name}"`,
          `"${a.asset_type}"`,
          `"${a.material_type}"`,
          a.latitude,
          a.longitude,
          `"${a.detection.defect_class}"`,
          a.detection.confidence_score,
          a.segmentation.crack_length_mm,
          a.segmentation.avg_width_mm,
          a.segmentation.max_width_mm,
          a.segmentation.crack_density_pct,
          a.assessment.severity_level,
          a.assessment.maintenance_priority,
          a.assessment.approval_status,
          `"${a.assessment.verified_by || engineerSignOffName || currentUser.full_name}"`,
        ].join(',')
      ),
    ];

    const blob = new Blob([csvLines.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `infrascan_structural_assessment_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    showNotice('Exported PostGIS & Quantitative Defect Assessment CSV.');
  };

  // Portfolio Aggregate Metrics
  const portfolioStats = useMemo(() => {
    const total = assets.length;
    const criticalOrSevere = assets.filter(
      (a) =>
        a.assessment.severity_level === 'Critical' || a.assessment.severity_level === 'Severe'
    ).length;
    const avgHealth =
      total > 0 ? Math.round(assets.reduce((acc, c) => acc + c.health_score, 0) / total) : 0;
    return { total, criticalOrSevere, avgHealth };
  }, [assets]);

  const activePermissions = ROLE_PERMISSIONS_MAP[currentUser.role];
  const qcPreviewAsset =
    assets.find((a) => a.asset_id === qcSelectedPresetId) || selectedAsset || assets[0];

  // ==========================================================================
  // VIEW 0: ENTERPRISE LOGIN & CREATE ACCOUNT PAGE
  // (No Example Assigned Users · Access Key 1234 Required ONLY for Administrator)
  // ==========================================================================
  if (!isAuthenticated) {
    const selectedRoleInfo = ROLE_PERMISSIONS_MAP[authSelectedRole];
    return (
      <div className="min-h-screen bg-[#F8FAFC] text-slate-900 flex flex-col justify-between">
        {/* Top Minimalist Header */}
        <header className="flex items-center justify-between gap-8 px-6 py-4 border-b border-slate-200 bg-white">
          <div className="flex items-center gap-2.5 shrink-0">
            <EnterpriseBrandLogo size="sm" />
            <span className="text-base font-bold tracking-tight text-slate-900 font-display whitespace-nowrap">
              InfraScan
            </span>
          </div>

          <div className="hidden md:flex items-center gap-6 text-xs text-slate-500">
            <span>YOLOv8/11 Detection</span>
            <span aria-hidden="true">·</span>
            <span>U-Net Segmentation</span>
            <span aria-hidden="true">·</span>
            <span>Grad-CAM XAI</span>
            <span aria-hidden="true">·</span>
            <span>PostGIS Spatial Inventory</span>
          </div>

          <div className="text-xs font-mono text-slate-500 whitespace-nowrap shrink-0">
            OAuth2 / JWT Portal
          </div>
        </header>

        {/* Main Centered Login / Create Account Split Layout */}
        <main className="flex-1 max-w-[1120px] w-full mx-auto px-6 py-10 flex items-center">
          <div className="w-full grid grid-cols-1 lg:grid-cols-12 gap-10 items-stretch">
            {/* Left 6 Columns: Enterprise Identity & RBAC Role Scope Matrix */}
            <div className="lg:col-span-6 flex flex-col justify-between py-2 space-y-8">
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <EnterpriseBrandLogo size="lg" />
                  <div>
                    <p className="text-xs font-medium text-blue-600">
                      Enterprise Civil Infrastructure AI Platform
                    </p>
                    <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 font-display">
                      InfraScan Assessment Portal
                    </h1>
                  </div>
                </div>

                <p className="text-sm text-slate-600 leading-relaxed">
                  Log in to your existing account or create a new operator account by choosing your
                  civil engineering role from the dropdown menu. Standard roles require only your
                  credentials; selecting the <strong>Administrator</strong> role requires the
                  Administrator Access Key.
                </p>
              </div>

              {/* Clean RBAC Role Permissions Directory (No Example Assigned Users) */}
              <div className="border border-slate-200 rounded-lg bg-white p-5 space-y-3">
                <div className="flex items-center justify-between text-xs border-b border-slate-100 pb-2.5">
                  <span className="font-semibold text-slate-800">
                    Available RBAC Operational Roles
                  </span>
                  <span className="font-mono text-slate-500">Session-Locked</span>
                </div>
                <div className="divide-y divide-slate-100 text-xs">
                  {(Object.keys(ROLE_PERMISSIONS_MAP) as RoleType[]).map((rKey) => {
                    const info = ROLE_PERMISSIONS_MAP[rKey];
                    const isSelected = authSelectedRole === rKey;
                    return (
                      <div
                        key={rKey}
                        className={`py-2.5 flex items-center justify-between gap-3 ${
                          isSelected ? 'text-blue-700 font-semibold' : 'text-slate-700'
                        }`}
                      >
                        <div>
                          <div>{rKey}</div>
                          <div className="text-[11px] text-slate-500 font-normal mt-0.5">
                            {info.summary}
                          </div>
                        </div>
                        <span className="font-mono text-[11px] text-slate-600 shrink-0">
                          {info.requiresAdminKey ? 'Requires Admin Key' : 'Standard Access'}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Security Audit Notice */}
              <div className="border border-slate-200 rounded-lg bg-slate-100/80 p-4 space-y-1.5">
                <div className="flex items-center gap-2 text-xs font-semibold text-slate-800">
                  <ShieldCheck className="w-4 h-4 text-blue-600 shrink-0" />
                  <span>Security & Compliance Audit Notice</span>
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">
                  All authentication sessions issue signed HMAC-SHA256 JSON Web Tokens (JWT). Your
                  chosen role is locked upon authentication and cannot be modified during an active
                  session without an authorized Administrator override key.
                </p>
              </div>
            </div>

            {/* Right 6 Columns: Log In / Create Account Card */}
            <div className="lg:col-span-6">
              <div className="border border-slate-200 rounded-xl bg-white p-6 sm:p-8 shadow-xs space-y-6">
                {/* Mode Switcher: Log In vs Create Account */}
                <div className="grid grid-cols-2 gap-1 p-1 bg-slate-100 border border-slate-200 rounded-lg">
                  <button
                    type="button"
                    onClick={() => {
                      setAuthMode('login');
                      setAuthError(null);
                    }}
                    className={`py-2 text-xs font-semibold rounded-md transition-colors flex items-center justify-center gap-1.5 cursor-pointer ${
                      authMode === 'login'
                        ? 'bg-white text-slate-900 shadow-2xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    <KeyRound className="w-3.5 h-3.5" />
                    <span>Log In</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setAuthMode('register');
                      setAuthError(null);
                    }}
                    className={`py-2 text-xs font-semibold rounded-md transition-colors flex items-center justify-center gap-1.5 cursor-pointer ${
                      authMode === 'register'
                        ? 'bg-white text-slate-900 shadow-2xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    <UserPlus className="w-3.5 h-3.5" />
                    <span>Create Account</span>
                  </button>
                </div>

                <div className="pb-3 border-b border-slate-200">
                  <h2 className="text-lg font-semibold text-slate-900">
                    {authMode === 'login'
                      ? 'Log In to InfraScan Portal'
                      : 'Create New Operator Account'}
                  </h2>
                  <p className="text-xs text-slate-500 mt-0.5">
                    {authMode === 'login'
                      ? 'Enter your credentials and active role to start a locked JWT session.'
                      : 'Register your profile and choose your operational role from the dropdown.'}
                  </p>
                </div>

                <form onSubmit={(e) => handleAuthSubmit(e, false)} className="space-y-4">
                  {/* Full Name (Shown in Create Account mode) */}
                  {authMode === 'register' && (
                    <div>
                      <label className="block text-xs font-medium text-slate-700 mb-1.5">
                        Full Name & Title
                      </label>
                      <input
                        type="text"
                        required
                        placeholder="Enter your full name"
                        value={authFullName}
                        onChange={(e) => setAuthFullName(e.target.value)}
                        className="w-full px-3 py-2 text-xs bg-white border border-slate-300 rounded-lg text-slate-900 focus:outline-none focus:border-blue-600"
                      />
                    </div>
                  )}

                  {/* Email Input */}
                  <div>
                    <label className="block text-xs font-medium text-slate-700 mb-1.5">
                      Enterprise Email Address
                    </label>
                    <input
                      type="email"
                      required
                      placeholder="name@organization.org"
                      value={authEmail}
                      onChange={(e) => setAuthEmail(e.target.value)}
                      className="w-full px-3 py-2 text-xs font-mono bg-white border border-slate-300 rounded-lg text-slate-900 focus:outline-none focus:border-blue-600"
                    />
                  </div>

                  {/* Password Input */}
                  <div>
                    <label className="block text-xs font-medium text-slate-700 mb-1.5">
                      Password
                    </label>
                    <input
                      type="password"
                      required
                      placeholder="Enter your password"
                      value={authPassword}
                      onChange={(e) => setAuthPassword(e.target.value)}
                      className="w-full px-3 py-2 text-xs font-mono bg-white border border-slate-300 rounded-lg text-slate-900 focus:outline-none focus:border-blue-600"
                    />
                  </div>

                  {/* Role Selector Dropdown (All 5 Roles, No Example User Names) */}
                  <div>
                    <label className="block text-xs font-medium text-slate-700 mb-1.5">
                      Select Active Role (Locked After Sign-In)
                    </label>
                    <select
                      value={authSelectedRole}
                      onChange={(e) => {
                        setAuthSelectedRole(e.target.value as RoleType);
                        setAuthError(null);
                      }}
                      className="w-full px-3 py-2.5 text-xs font-medium bg-slate-50 border border-slate-300 rounded-lg text-slate-900 focus:outline-none focus:border-blue-600 focus:bg-white transition-colors"
                    >
                      <option value="Civil Engineer">Civil Engineer</option>
                      <option value="Infrastructure Manager">Infrastructure Manager</option>
                      <option value="Field Inspector">Field Inspector</option>
                      <option value="Maintenance Engineer">Maintenance Engineer</option>
                      <option value="Administrator">Administrator (Requires Access Key)</option>
                    </select>
                    <p className="text-[11px] text-slate-500 mt-1.5">
                      Role Permissions: {selectedRoleInfo.summary}
                    </p>
                  </div>

                  {/* Conditional Administrator Access Key Input (Shown ONLY when Administrator is chosen) */}
                  {authSelectedRole === 'Administrator' && (
                    <div className="p-3.5 rounded-lg bg-amber-50/70 border border-amber-200 space-y-2">
                      <div className="flex items-center justify-between">
                        <label className="text-xs font-semibold text-amber-900 flex items-center gap-1.5">
                          <Lock className="w-3.5 h-3.5 text-amber-700" />
                          <span>Administrator Access Key Required</span>
                        </label>
                        <span className="text-[11px] font-mono text-amber-700">Restricted</span>
                      </div>
                      <input
                        type="password"
                        required
                        placeholder="Enter Administrator Access Key"
                        value={authAdminAccessKey}
                        onChange={(e) => setAuthAdminAccessKey(e.target.value)}
                        className="w-full px-3 py-2 text-xs font-mono bg-white border border-amber-300 rounded-md text-slate-900 focus:outline-none focus:border-amber-600"
                      />
                    </div>
                  )}

                  {authError && (
                    <div className="p-3 rounded-lg bg-red-50 border border-red-200 text-xs text-red-700 font-medium space-y-2">
                      <div>{authError}</div>
                      {authError.toLowerCase().includes('user not found') && (
                        <button
                          type="button"
                          onClick={() => {
                            setAuthMode('register');
                            setAuthError(null);
                          }}
                          className="inline-flex items-center gap-1 text-xs font-semibold text-blue-700 hover:text-blue-800 underline cursor-pointer"
                        >
                          <span>Create an account first</span>
                          <ArrowRight className="w-3.5 h-3.5" />
                        </button>
                      )}
                      {authError.toLowerCase().includes('already exists') && (
                        <button
                          type="button"
                          onClick={() => {
                            setAuthMode('login');
                            setAuthError(null);
                          }}
                          className="inline-flex items-center gap-1 text-xs font-semibold text-blue-700 hover:text-blue-800 underline cursor-pointer"
                        >
                          <span>Switch to Log In</span>
                          <ArrowRight className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  )}

                  {/* Primary Submit Button */}
                  <button
                    type="submit"
                    disabled={isAuthenticating}
                    className="w-full py-2.5 px-4 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors flex items-center justify-center gap-2 cursor-pointer"
                  >
                    <span>
                      {isAuthenticating
                        ? 'Authenticating...'
                        : authMode === 'register'
                        ? `Create Account as ${authSelectedRole}`
                        : `Log In as ${authSelectedRole}`}
                    </span>
                    <ArrowRight className="w-4 h-4" />
                  </button>
                </form>

                {/* Divider */}
                <div className="relative flex py-1 items-center">
                  <div className="flex-grow border-t border-slate-200" />
                  <span className="shrink mx-3 text-[11px] text-slate-400">
                    Or Federated Single Sign-On
                  </span>
                  <div className="flex-grow border-t border-slate-200" />
                </div>

                {/* Enterprise SSO Login Button */}
                <button
                  type="button"
                  onClick={() => handleAuthSubmit(undefined, true)}
                  className="w-full py-2.5 px-4 text-xs font-medium text-slate-800 bg-slate-100 hover:bg-slate-200/80 border border-slate-300 rounded-lg transition-colors flex items-center justify-center gap-2 cursor-pointer"
                >
                  <Building2 className="w-4 h-4 text-slate-600" />
                  <span>Continue with Enterprise SAML 2.0 / GovCloud SSO</span>
                </button>
              </div>
            </div>
          </div>
        </main>

        {/* Quiet Footer */}
        <footer className="border-t border-slate-200 bg-white py-4 px-6 text-xs text-slate-500">
          <div className="max-w-[1120px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
            <div>InfraScan Civil Infrastructure AI Assessment System · FastAPI + React + PostGIS</div>
            <div className="font-mono text-[11px]">OAuth2 Password Bearer · JWT HS256</div>
          </div>
        </footer>
      </div>
    );
  }

  // ==========================================================================
  // AUTHENTICATED MAIN WORKSPACE (ACTIVE ROLE LOCKED UNLESS ADMIN OVERRIDE)
  // ==========================================================================
  return (
    <div className="min-h-screen bg-[#F8FAFC] text-slate-900 flex flex-col">
      {/* =====================================================================
          TOP BAR CONTRACT (Strictly 3 Zones separated by gap-8)
      ===================================================================== */}
      <header className="flex items-center justify-between gap-8 px-6 py-3.5 border-b border-slate-200 bg-white sticky top-0 z-30">
        {/* Zone 1: Single text element wordmark */}
        <a
          href="#top"
          onClick={(e) => {
            e.preventDefault();
            setActiveWorkspace('assets');
          }}
          className="text-base font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0 font-display"
        >
          InfraScan
        </a>

        {/* Zone 2: 5 Clear Single-Line Navigation Links */}
        <nav className="flex items-center gap-6 text-sm font-medium text-slate-600 overflow-x-auto">
          <button
            type="button"
            onClick={() => setActiveWorkspace('assets')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 cursor-pointer ${
              activeWorkspace === 'assets'
                ? 'text-blue-600 border-blue-600 font-semibold'
                : 'border-transparent hover:text-slate-900'
            }`}
          >
            1. Asset Hub
          </button>
          <button
            type="button"
            onClick={() => setActiveWorkspace('qc')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 cursor-pointer ${
              activeWorkspace === 'qc'
                ? 'text-blue-600 border-blue-600 font-semibold'
                : 'border-transparent hover:text-slate-900'
            }`}
          >
            2. QC Workbench
          </button>
          <button
            type="button"
            onClick={() => setActiveWorkspace('inspection')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 cursor-pointer ${
              activeWorkspace === 'inspection'
                ? 'text-blue-600 border-blue-600 font-semibold'
                : 'border-transparent hover:text-slate-900'
            }`}
          >
            3. AI Inspection
          </button>
          <button
            type="button"
            onClick={() => setActiveWorkspace('map')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 cursor-pointer ${
              activeWorkspace === 'map'
                ? 'text-blue-600 border-blue-600 font-semibold'
                : 'border-transparent hover:text-slate-900'
            }`}
          >
            4. Spatial Map
          </button>
          <button
            type="button"
            onClick={() => setActiveWorkspace('reports')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 cursor-pointer ${
              activeWorkspace === 'reports'
                ? 'text-blue-600 border-blue-600 font-semibold'
                : 'border-transparent hover:text-slate-900'
            }`}
          >
            5. Analytics & Reports
          </button>
        </nav>

        {/* Zone 3: 1 Primary Action */}
        <div className="flex items-center shrink-0">
          <button
            type="button"
            onClick={() => setIsAuthenticated(false)}
            className="px-3.5 py-1.5 text-xs font-medium text-slate-700 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 rounded-md transition-colors whitespace-nowrap shrink-0 flex items-center gap-1.5 cursor-pointer"
          >
            <LogOut className="w-3.5 h-3.5" />
            <span>Sign Out</span>
          </button>
        </div>
      </header>

      {/* =====================================================================
          SUB-HEADER: LOCKED ACTIVE ROLE DROPDOWN (REQUIRES ADMIN OVERRIDE 1234)
      ===================================================================== */}
      <section className="border-b border-slate-200 bg-slate-100/70 px-6 py-2.5">
        <div className="max-w-[1440px] mx-auto flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3">
          {/* Left: Locked Active Role Dropdown (No Example User Names) */}
          <div className="flex flex-wrap items-center gap-3 text-xs">
            <div className="flex items-center gap-1.5 text-slate-700">
              <Lock className="w-3.5 h-3.5 text-amber-600 shrink-0" />
              <span className="font-semibold text-slate-900">Locked Active Role:</span>
            </div>

            <select
              value={currentUser.role}
              onChange={(e) => handleRoleDropdownSelectInWorkspace(e.target.value as RoleType)}
              className="px-2.5 py-1 text-xs font-semibold bg-white border border-slate-300 rounded-md text-slate-900 focus:outline-none focus:border-blue-600"
              title="Changing the active role after sign-in requires Administrator Override Key"
            >
              <option value="Civil Engineer">Civil Engineer</option>
              <option value="Infrastructure Manager">Infrastructure Manager</option>
              <option value="Field Inspector">Field Inspector</option>
              <option value="Maintenance Engineer">Maintenance Engineer</option>
              <option value="Administrator">Administrator</option>
            </select>

            <span className="text-[11px] text-slate-500">
              Operator: <span className="font-medium text-slate-800">{currentUser.full_name}</span>{' '}
              · Admin Override Key required to change role
            </span>
          </div>

          {/* Right: Quick Stats Summary + FastAPI Blueprint Drawer Trigger */}
          <div className="flex flex-wrap items-center gap-5 text-xs">
            <div className="flex items-center gap-1.5">
              <span className="text-slate-500">Assets:</span>
              <span className="font-mono font-semibold text-slate-900 tabular-nums">
                {portfolioStats.total}
              </span>
            </div>
            <span className="text-slate-300" aria-hidden="true">
              /
            </span>
            <div className="flex items-center gap-1.5">
              <span className="text-slate-500">Critical & Severe:</span>
              <span className="font-mono font-semibold text-red-600 tabular-nums">
                {portfolioStats.criticalOrSevere}
              </span>
            </div>
            <span className="text-slate-300" aria-hidden="true">
              /
            </span>
            <div className="flex items-center gap-1.5">
              <span className="text-slate-500">Mean Health:</span>
              <span className="font-mono font-semibold text-emerald-700 tabular-nums">
                {portfolioStats.avgHealth}/100
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* Notification Banner */}
      {toastMessage && (
        <div className="bg-white border-b border-slate-200 px-6 py-2.5 shadow-2xs">
          <div className="max-w-[1440px] mx-auto flex items-center justify-between text-xs">
            <div className="flex items-center gap-2">
              {toastMessage.type === 'error' ? (
                <XCircle className="w-4 h-4 text-red-600 shrink-0" />
              ) : (
                <CheckCircle2 className="w-4 h-4 text-blue-600 shrink-0" />
              )}
              <span
                className={
                  toastMessage.type === 'error'
                    ? 'text-red-700 font-medium'
                    : 'text-slate-800 font-medium'
                }
              >
                {toastMessage.text}
              </span>
            </div>
            <button
              type="button"
              onClick={() => setToastMessage(null)}
              className="text-slate-500 hover:text-slate-800 whitespace-nowrap shrink-0"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* =====================================================================
          MAIN VIEWPORT CONTAINER (1440px Desktop Baseline — Crisp Light Mode)
      ===================================================================== */}
      <main className="flex-1 max-w-[1440px] w-full mx-auto px-6 py-8">
        {/* ===================================================================
            WORKSPACE VIEW 1: INFRASTRUCTURE ASSET HUB
        =================================================================== */}
        {activeWorkspace === 'assets' && (
          <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 pb-5 border-b border-slate-200">
              <div>
                <p className="text-xs font-medium text-blue-600 mb-1">
                  01. Civil Infrastructure Registry · PostGIS Spatial Inventory
                </p>
                <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">
                  Infrastructure Asset Hub
                </h1>
                <p className="text-sm text-slate-600 mt-1 max-w-2xl">
                  Select any civil structure below to inspect its YOLOv11 bounding boxes, U-Net
                  segmentation masks, or historical fracture progression.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-3 shrink-0">
                <div className="flex items-center gap-1 p-1 bg-slate-100 border border-slate-200 rounded-lg">
                  <button
                    type="button"
                    onClick={() => setAssetLayout('table')}
                    className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      assetLayout === 'table'
                        ? 'bg-white text-slate-900 shadow-2xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Table View
                  </button>
                  <button
                    type="button"
                    onClick={() => setAssetLayout('grid')}
                    className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      assetLayout === 'grid'
                        ? 'bg-white text-slate-900 shadow-2xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Card Grid
                  </button>
                  <button
                    type="button"
                    onClick={() => setAssetLayout('priority')}
                    className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      assetLayout === 'priority'
                        ? 'bg-white text-slate-900 shadow-2xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Priority Queue
                  </button>
                </div>

                <button
                  type="button"
                  onClick={() => {
                    if (!activePermissions.canRegisterAsset) {
                      showNotice(
                        `RBAC Restriction: Locked role "${currentUser.role}" cannot register new infrastructure assets. Administrator override required.`,
                        'error'
                      );
                      return;
                    }
                    setShowNewAssetModal(true);
                  }}
                  className="flex items-center gap-1.5 px-4 py-2 text-xs font-semibold bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                >
                  <Plus className="w-3.5 h-3.5" />
                  <span>Register Asset</span>
                </button>

                <button
                  type="button"
                  onClick={() => setActiveWorkspace('qc')}
                  className="flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                >
                  <span>Next: QC Workbench</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* Filter & Search Bar */}
            <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center bg-white border border-slate-200 rounded-lg p-4">
              <div className="md:col-span-4 relative">
                <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search by structure name, defect class, or department..."
                  className="w-full pl-9 pr-3 py-2 text-xs bg-slate-50 border border-slate-200 rounded-md text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-blue-600 focus:bg-white"
                />
              </div>

              <div className="md:col-span-3 flex items-center gap-2">
                <label className="text-xs text-slate-500 whitespace-nowrap shrink-0">
                  Type:
                </label>
                <select
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value)}
                  className="w-full px-2.5 py-2 text-xs bg-slate-50 border border-slate-200 rounded-md text-slate-800 focus:outline-none focus:border-blue-600"
                >
                  <option value="All">All Asset Types ({assets.length})</option>
                  <option value="Bridge">Bridge</option>
                  <option value="Pavement">Pavement</option>
                  <option value="Vertical Building">Vertical Building</option>
                  <option value="Tunnel">Tunnel</option>
                  <option value="Hydraulic Dam">Hydraulic Dam</option>
                </select>
              </div>

              <div className="md:col-span-3 flex items-center gap-2">
                <label className="text-xs text-slate-500 whitespace-nowrap shrink-0">
                  Severity:
                </label>
                <select
                  value={filterSeverity}
                  onChange={(e) => setFilterSeverity(e.target.value)}
                  className="w-full px-2.5 py-2 text-xs bg-slate-50 border border-slate-200 rounded-md text-slate-800 focus:outline-none focus:border-blue-600"
                >
                  <option value="All">All Severity Levels</option>
                  <option value="Critical">Critical</option>
                  <option value="Severe">Severe</option>
                  <option value="Moderate">Moderate</option>
                  <option value="Minor">Minor</option>
                </select>
              </div>

              <div className="md:col-span-2 flex items-center gap-2">
                <select
                  value={filterPriority}
                  onChange={(e) => setFilterPriority(e.target.value)}
                  className="w-full px-2.5 py-2 text-xs bg-slate-50 border border-slate-200 rounded-md text-slate-800 focus:outline-none focus:border-blue-600"
                >
                  <option value="All">All Priorities</option>
                  <option value="Immediate">Immediate</option>
                  <option value="Urgent">Urgent</option>
                  <option value="Scheduled">Scheduled</option>
                  <option value="Routine">Routine</option>
                </select>
              </div>
            </div>

            {isLoading ? (
              <div className="border border-slate-200 rounded-lg overflow-hidden bg-white">
                {[1, 2, 3, 4, 5].map((n) => (
                  <div
                    key={n}
                    className="h-14 border-b border-slate-100 px-6 flex items-center justify-between animate-pulse"
                  >
                    <div className="h-3 w-48 bg-slate-200 rounded" />
                    <div className="h-3 w-32 bg-slate-200 rounded" />
                    <div className="h-3 w-24 bg-slate-200 rounded" />
                  </div>
                ))}
              </div>
            ) : filteredAssets.length === 0 ? (
              <div className="border border-slate-200 rounded-lg bg-white p-12 text-center">
                <p className="text-sm font-medium text-slate-800">
                  No civil infrastructure assets match the active filter criteria.
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  Reset your structure type, severity, or search query to inspect the full PostGIS
                  inventory.
                </p>
                <button
                  type="button"
                  onClick={() => {
                    setFilterType('All');
                    setFilterSeverity('All');
                    setFilterPriority('All');
                    setSearchQuery('');
                  }}
                  className="mt-4 px-4 py-2 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-md transition-colors whitespace-nowrap shrink-0"
                >
                  Reset Filters
                </button>
              </div>
            ) : assetLayout === 'table' ? (
              <div className="border border-slate-200 rounded-lg bg-white overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50/80 text-[11px] font-semibold text-slate-500">
                      <th className="py-3.5 px-5">Civil Infrastructure Asset</th>
                      <th className="py-3.5 px-4">Type · Material · Year</th>
                      <th className="py-3.5 px-4">YOLOv11 Defect Class</th>
                      <th className="py-3.5 px-4 text-right">Length · Max Width</th>
                      <th className="py-3.5 px-4 text-right">Crack Density</th>
                      <th className="py-3.5 px-4 text-right">Health Score</th>
                      <th className="py-3.5 px-4">Severity · Priority</th>
                      <th className="py-3.5 px-5 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200/80 text-xs">
                    {filteredAssets.map((asset) => {
                      const sev = getSeverityMeta(asset.assessment.severity_level);
                      const SevIcon = sev.icon;
                      return (
                        <tr
                          key={asset.asset_id}
                          onClick={() => setSelectedAssetId(asset.asset_id)}
                          className={`transition-colors cursor-pointer ${
                            selectedAssetId === asset.asset_id
                              ? 'bg-blue-50/60'
                              : 'hover:bg-slate-50'
                          }`}
                        >
                          <td className="py-3.5 px-5">
                            <div className="font-semibold text-slate-900">{asset.asset_name}</div>
                            <div className="text-[11px] text-slate-500 mt-0.5">
                              {asset.responsible_department} ·{' '}
                              <span className="font-mono tabular-nums">
                                {asset.latitude.toFixed(4)}°N, {asset.longitude.toFixed(4)}°W
                              </span>
                            </div>
                          </td>

                          <td className="py-3.5 px-4 text-slate-700">
                            <div className="font-medium">{asset.asset_type}</div>
                            <div className="text-[11px] text-slate-500 mt-0.5">
                              {asset.material_type} · Built {asset.construction_year}
                            </div>
                          </td>

                          <td className="py-3.5 px-4">
                            <div className="font-mono font-medium text-slate-900">
                              {asset.detection.defect_class}
                            </div>
                            <div className="text-[11px] font-mono text-slate-500 tabular-nums mt-0.5">
                              Conf: {(asset.detection.confidence_score * 100).toFixed(1)}% ·
                              Laplacian: {asset.image.blur_laplacian_var}
                            </div>
                          </td>

                          <td className="py-3.5 px-4 text-right font-mono tabular-nums">
                            <div className="text-slate-900 font-medium">
                              {asset.segmentation.crack_length_mm.toFixed(1)} mm
                            </div>
                            <div className="text-[11px] text-slate-500 mt-0.5">
                              Max: {asset.segmentation.max_width_mm.toFixed(2)} mm
                            </div>
                          </td>

                          <td className="py-3.5 px-4 text-right font-mono tabular-nums">
                            <span className="text-slate-900 font-semibold">
                              {asset.segmentation.crack_density_pct.toFixed(1)}%
                            </span>
                          </td>

                          <td className="py-3.5 px-4 text-right font-mono tabular-nums">
                            <span
                              className={`font-semibold ${
                                asset.health_score < 50
                                  ? 'text-red-700'
                                  : asset.health_score < 70
                                  ? 'text-amber-700'
                                  : 'text-emerald-700'
                              }`}
                            >
                              {asset.health_score}/100
                            </span>
                          </td>

                          <td className="py-3.5 px-4">
                            <div
                              className={`flex items-center gap-1.5 font-semibold ${sev.textClass}`}
                            >
                              <SevIcon className="w-3.5 h-3.5 shrink-0" />
                              <span>
                                {asset.assessment.severity_level} ·{' '}
                                {asset.assessment.maintenance_priority}
                              </span>
                            </div>
                            <div className="text-[11px] text-slate-500 mt-0.5">
                              {asset.assessment.approval_status}
                            </div>
                          </td>

                          <td className="py-3.5 px-5 text-right">
                            <div
                              className="inline-flex items-center gap-2"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <button
                                type="button"
                                onClick={() => handleOpenHistory(asset.asset_id)}
                                className="px-2.5 py-1 text-xs font-medium text-slate-700 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                              >
                                History
                              </button>
                              <button
                                type="button"
                                onClick={() => {
                                  setSelectedAssetId(asset.asset_id);
                                  setActiveWorkspace('inspection');
                                }}
                                className="px-2.5 py-1 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded transition-colors whitespace-nowrap shrink-0 flex items-center gap-1 cursor-pointer"
                              >
                                <span>Inspect AI</span>
                                <ArrowUpRight className="w-3 h-3" />
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredAssets.map((asset) => {
                  const sev = getSeverityMeta(asset.assessment.severity_level);
                  const SevIcon = sev.icon;
                  return (
                    <div
                      key={asset.asset_id}
                      className="border border-slate-200 rounded-lg bg-white overflow-hidden flex flex-col justify-between hover:border-slate-300 transition-colors"
                    >
                      <div>
                        <div className="relative h-48 bg-slate-900 border-b border-slate-200 overflow-hidden">
                          <ResilientInfrastructureImage
                            src={asset.image.storage_url}
                            alt={asset.asset_name}
                            className="w-full h-full object-cover"
                          />
                          <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/25 to-transparent" />
                          <div className="absolute bottom-3 left-4 right-4 flex items-end justify-between">
                            <div>
                              <p className="text-[11px] text-slate-200">
                                {asset.asset_type} · Built {asset.construction_year}
                              </p>
                              <h3 className="text-base font-semibold text-white">
                                {asset.asset_name}
                              </h3>
                            </div>
                            <span className="font-mono text-xs font-semibold text-white tabular-nums">
                              Health {asset.health_score}/100
                            </span>
                          </div>
                        </div>

                        <div className="p-5 space-y-4">
                          <div className="flex items-center justify-between text-xs">
                            <span className="font-mono font-semibold text-slate-900">
                              {asset.detection.defect_class}
                            </span>
                            <span
                              className={`flex items-center gap-1 font-semibold ${sev.textClass}`}
                            >
                              <SevIcon className="w-3.5 h-3.5" />
                              <span>
                                {asset.assessment.severity_level} ·{' '}
                                {asset.assessment.maintenance_priority}
                              </span>
                            </span>
                          </div>

                          <div className="grid grid-cols-3 gap-2 pt-3 border-t border-slate-100 text-xs font-mono tabular-nums">
                            <div>
                              <span className="block text-[10px] font-sans text-slate-500">
                                Crack Length
                              </span>
                              <span className="text-slate-900 font-semibold">
                                {asset.segmentation.crack_length_mm} mm
                              </span>
                            </div>
                            <div>
                              <span className="block text-[10px] font-sans text-slate-500">
                                Max Aperture
                              </span>
                              <span className="text-slate-900 font-semibold">
                                {asset.segmentation.max_width_mm} mm
                              </span>
                            </div>
                            <div>
                              <span className="block text-[10px] font-sans text-slate-500">
                                Surface Density
                              </span>
                              <span className="text-slate-900 font-semibold">
                                {asset.segmentation.crack_density_pct}%
                              </span>
                            </div>
                          </div>

                          <p className="text-xs text-slate-600 line-clamp-2">
                            {asset.assessment.engineer_comments}
                          </p>
                        </div>
                      </div>

                      <div className="px-5 py-3 border-t border-slate-200 flex items-center justify-between bg-slate-50">
                        <button
                          type="button"
                          onClick={() => handleOpenHistory(asset.asset_id)}
                          className="text-xs font-medium text-slate-600 hover:text-slate-900 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                        >
                          View History (3)
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setSelectedAssetId(asset.asset_id);
                            setActiveWorkspace('inspection');
                          }}
                          className="px-3 py-1.5 text-xs font-semibold bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                        >
                          Open AI Analysis
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {assetLayout === 'priority' && (
              <div className="border border-slate-200 rounded-lg bg-white overflow-hidden">
                <div className="px-5 py-4 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-2 bg-slate-50/70">
                  <div>
                    <h2 className="text-sm font-semibold text-slate-900">
                      Maintenance Prioritization & Field Dispatch Queue
                    </h2>
                    <p className="text-xs text-slate-500">
                      Risk-ranked work orders sorted by severity, SLA window, and maximum crack aperture.
                    </p>
                  </div>
                  <span className="text-xs font-mono text-slate-600 tabular-nums">
                    Active Queue: {filteredAssets.length} Work Orders
                  </span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 text-xs font-medium text-slate-500 bg-slate-50">
                        <th className="py-3 px-4">Work Order</th>
                        <th className="py-3 px-4">Infrastructure Asset</th>
                        <th className="py-3 px-4">Severity & Priority</th>
                        <th className="py-3 px-4 text-right">Max Aperture</th>
                        <th className="py-3 px-4 text-right">SLA Window</th>
                        <th className="py-3 px-4 text-right">Est. Budget</th>
                        <th className="py-3 px-4">Assigned Team</th>
                        <th className="py-3 px-4">Workflow Stage</th>
                        <th className="py-3 px-4 text-right">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200/70 text-xs">
                      {[...filteredAssets]
                        .sort((a, b) => {
                          const rank: Record<string, number> = {
                            Immediate: 4,
                            Urgent: 3,
                            Scheduled: 2,
                            Routine: 1,
                          };
                          const diff =
                            (rank[b.assessment.maintenance_priority] || 0) -
                            (rank[a.assessment.maintenance_priority] || 0);
                          return diff !== 0
                            ? diff
                            : b.segmentation.max_width_mm - a.segmentation.max_width_mm;
                        })
                        .map((asset) => {
                          const sev = getSeverityMeta(asset.assessment.severity_level);
                          const wf = asset.maintenance_workflow;
                          return (
                            <tr key={asset.asset_id} className="hover:bg-slate-50 transition-colors">
                              <td className="py-3.5 px-4 font-mono font-semibold text-slate-800 whitespace-nowrap">
                                {wf?.work_order_id || `WO-2026-${1040 + asset.asset_id}`}
                              </td>
                              <td className="py-3.5 px-4">
                                <div className="font-semibold text-slate-900">{asset.asset_name}</div>
                                <div className="text-[11px] text-slate-500">
                                  {asset.asset_type} · {asset.detection.defect_class}
                                </div>
                              </td>
                              <td className="py-3.5 px-4 whitespace-nowrap">
                                <div className={`font-semibold ${sev.textClass}`}>
                                  {asset.assessment.severity_level}
                                </div>
                                <div className="text-[11px] font-mono text-slate-600">
                                  {asset.assessment.maintenance_priority}
                                </div>
                              </td>
                              <td className="py-3.5 px-4 text-right font-mono tabular-nums text-slate-900">
                                {asset.segmentation.max_width_mm.toFixed(2)} mm
                              </td>
                              <td className="py-3.5 px-4 text-right font-mono tabular-nums text-slate-700">
                                {wf?.sla_days ?? 30}d
                              </td>
                              <td className="py-3.5 px-4 text-right font-mono tabular-nums text-slate-900">
                                ${(wf?.estimated_cost_usd ?? 25000).toLocaleString()}
                              </td>
                              <td className="py-3.5 px-4 text-slate-700">
                                {wf?.assigned_team || asset.responsible_department}
                              </td>
                              <td className="py-3.5 px-4">
                                <select
                                  value={wf?.workflow_stage || 'Triaged'}
                                  onChange={(e) =>
                                    handleUpdateWorkflowStage(
                                      asset.asset_id,
                                      e.target.value as
                                        | 'Triaged'
                                        | 'Scheduled'
                                        | 'Crew Dispatched'
                                        | 'In Remediation'
                                        | 'Verified Closed'
                                    )
                                  }
                                  className="px-2.5 py-1.5 text-xs bg-white border border-slate-300 rounded text-slate-900 font-medium focus:outline-none focus:border-blue-600"
                                >
                                  <option value="Triaged">Triaged</option>
                                  <option value="Scheduled">Scheduled</option>
                                  <option value="Crew Dispatched">Crew Dispatched</option>
                                  <option value="In Remediation">In Remediation</option>
                                  <option value="Verified Closed">Verified Closed</option>
                                </select>
                              </td>
                              <td className="py-3.5 px-4 text-right">
                                <button
                                  type="button"
                                  onClick={() => {
                                    setSelectedAssetId(asset.asset_id);
                                    setActiveWorkspace('inspection');
                                    setInspectionSubTab('viewport');
                                  }}
                                  className="px-2.5 py-1.5 text-xs font-medium text-blue-600 hover:bg-blue-50 rounded transition-colors whitespace-nowrap cursor-pointer"
                                >
                                  Inspect
                                </button>
                              </td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ===================================================================
            WORKSPACE VIEW 2: IMAGE QUALITY & PREPROCESSING WORKBENCH
        =================================================================== */}
        {activeWorkspace === 'qc' && qcPreviewAsset && (
          <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 pb-5 border-b border-slate-200">
              <div>
                <p className="text-xs font-medium text-blue-600 mb-1">
                  02. Pre-Inference Quality Control Gating · Laplacian Variance & Scale Calibration
                </p>
                <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">
                  Image Quality & Preprocessing Workbench
                </h1>
                <p className="text-sm text-slate-600 mt-1 max-w-2xl">
                  Automated optical sharpness verification, illumination dynamic range check, CLAHE
                  local contrast equalization, and photogrammetric scale calibration.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-3 shrink-0">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="px-3.5 py-2 text-xs font-medium bg-white hover:bg-slate-50 text-slate-800 border border-slate-300 rounded-lg transition-colors whitespace-nowrap shrink-0 flex items-center gap-1.5 cursor-pointer"
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span>Upload Local Frame</span>
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleFileUpload(f);
                  }}
                />
                <button
                  type="button"
                  disabled={!qcResult?.qc_passed}
                  onClick={handleIngestFrameToPipeline}
                  className={`px-4 py-2 text-xs font-semibold rounded-lg transition-colors whitespace-nowrap shrink-0 flex items-center gap-1.5 ${
                    qcResult?.qc_passed
                      ? 'bg-blue-600 hover:bg-blue-700 text-white cursor-pointer'
                      : 'bg-slate-200 text-slate-400 cursor-not-allowed'
                  }`}
                >
                  <span>Ingest to YOLOv11 + U-Net</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
              <div className="lg:col-span-7 space-y-6">
                <div
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    const f = e.dataTransfer.files?.[0];
                    if (f) handleFileUpload(f);
                  }}
                  className="border border-dashed border-slate-300 hover:border-blue-600 rounded-lg bg-white p-5 transition-colors"
                >
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-200">
                    <div>
                      <h3 className="text-sm font-semibold text-slate-900">
                        Source Optical Capture Frame
                      </h3>
                      <p className="text-xs text-slate-500 mt-0.5">
                        Drag and drop a high-resolution structural image or select a calibrated
                        survey capture below.
                      </p>
                    </div>

                    <select
                      value={qcSelectedPresetId}
                      onChange={(e) => {
                        setQcCustomImageSrc(null);
                        setQcSelectedPresetId(Number(e.target.value));
                      }}
                      className="px-3 py-1.5 text-xs bg-slate-50 border border-slate-300 rounded-md text-slate-900 font-medium"
                    >
                      {assets.map((a) => (
                        <option key={a.asset_id} value={a.asset_id}>
                          Asset #{a.asset_id}: {a.asset_name}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="flex flex-wrap items-center justify-between gap-4 py-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-600">Focus Test:</span>
                      <div className="flex items-center gap-1 p-1 bg-slate-100 border border-slate-200 rounded-md">
                        <button
                          type="button"
                          onClick={() => setQcSimulatedBlurMode('sharp')}
                          className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                            qcSimulatedBlurMode === 'sharp'
                              ? 'bg-emerald-600 text-white'
                              : 'text-slate-600 hover:text-slate-900'
                          }`}
                        >
                          Calibrated Sharp Capture
                        </button>
                        <button
                          type="button"
                          onClick={() => setQcSimulatedBlurMode('motion_blur')}
                          className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                            qcSimulatedBlurMode === 'motion_blur'
                              ? 'bg-red-600 text-white'
                              : 'text-slate-600 hover:text-slate-900'
                          }`}
                        >
                          Simulate Defocus Blur (Var &lt; 100)
                        </button>
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={() => setQcClaheEnabled((v) => !v)}
                      className={`px-3 py-1.5 text-xs font-medium rounded border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                        qcClaheEnabled
                          ? 'bg-blue-50 border-blue-300 text-blue-700'
                          : 'bg-slate-50 border-slate-200 text-slate-600'
                      }`}
                    >
                      CLAHE Histogram Equalization: {qcClaheEnabled ? 'ACTIVE' : 'BYPASSED'}
                    </button>
                  </div>

                  <div className="relative rounded-md overflow-hidden border border-slate-200 bg-slate-900 aspect-[4/3]">
                    <ResilientInfrastructureImage
                      src={qcCustomImageSrc || qcPreviewAsset.image.storage_url}
                      alt={qcPreviewAsset.asset_name}
                      className="w-full h-full object-cover transition-all duration-200"
                      style={{
                        filter: `${
                          qcSimulatedBlurMode === 'motion_blur' ? 'blur(5px)' : 'none'
                        } ${qcClaheEnabled ? 'contrast(1.22) brightness(1.04)' : 'none'}`,
                      }}
                    />

                    <div className="absolute inset-0 grid grid-cols-2 grid-rows-2 pointer-events-none">
                      {[
                        { id: 'T00 (NW)', pos: '0,0 → 1024,768' },
                        { id: 'T01 (NE)', pos: '1024,0 → 2048,768' },
                        { id: 'T10 (SW)', pos: '0,768 → 1024,1536' },
                        { id: 'T11 (SE)', pos: '1024,768 → 2048,1536' },
                      ].map((tile) => (
                        <div
                          key={tile.id}
                          className="border border-white/40 flex flex-col justify-between p-3 bg-gradient-to-b from-black/50 via-transparent to-black/50"
                        >
                          <div className="flex items-center justify-between text-[11px] font-mono text-white">
                            <span>Tile {tile.id}</span>
                            <span>{tile.pos}</span>
                          </div>
                          <div className="text-[11px] font-mono text-sky-300">
                            {qcClaheEnabled ? 'CLAHE 8×8 Clip=2.0' : 'Raw Luminance'}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div className="lg:col-span-5 space-y-6">
                {qcResult && (
                  <div className="border border-slate-200 rounded-lg bg-white p-5 space-y-5">
                    <div className="flex items-start justify-between gap-4 pb-4 border-b border-slate-200">
                      <div>
                        <span className="text-xs font-mono text-slate-500">
                          POST /api/v1/preprocess/validate
                        </span>
                        <h3 className="text-base font-semibold text-slate-900 mt-0.5">
                          Quality Control Gate Status
                        </h3>
                      </div>
                      <div
                        className={`flex items-center gap-1.5 text-xs font-mono font-semibold ${
                          qcResult.qc_passed ? 'text-emerald-700' : 'text-red-600'
                        }`}
                      >
                        {qcResult.qc_passed ? (
                          <CheckCircle2 className="w-4 h-4" />
                        ) : (
                          <XCircle className="w-4 h-4" />
                        )}
                        <span>{qcResult.qc_passed ? 'QC GATE PASSED' : 'QC GATE REJECTED'}</span>
                      </div>
                    </div>

                    <div className="space-y-3 text-xs">
                      <div className="flex items-center justify-between py-2 border-b border-slate-100">
                        <span className="text-slate-600">
                          Laplacian Variance
                        </span>
                        <span
                          className={`font-mono font-semibold tabular-nums ${
                            qcResult.laplacian_variance >= 100
                              ? 'text-emerald-700'
                              : 'text-red-600'
                          }`}
                        >
                          {qcResult.laplacian_variance} (Threshold ≥ 100.0)
                        </span>
                      </div>

                      <div className="flex items-center justify-between py-2 border-b border-slate-100">
                        <span className="text-slate-600">Focus Diagnostic</span>
                        <span className="font-mono text-slate-900 font-medium">
                          {qcResult.blur_status}
                        </span>
                      </div>

                      <div className="flex items-center justify-between py-2 border-b border-slate-100">
                        <span className="text-slate-600">Mean Luminance / Brightness</span>
                        <span className="font-mono text-slate-900 tabular-nums">
                          {qcResult.avg_brightness} / 255 ({qcResult.brightness_status})
                        </span>
                      </div>

                      <div className="flex items-center justify-between py-2 border-b border-slate-100">
                        <span className="text-slate-600">Sensor Resolution Check</span>
                        <span className="font-mono text-slate-900 tabular-nums">
                          {qcResult.dimensions} · SHA256 {qcResult.image_hash}
                        </span>
                      </div>

                      <div className="flex items-center justify-between py-2">
                        <span className="text-slate-600">CLAHE Normalization</span>
                        <span className="font-mono text-slate-900">
                          {qcResult.clahe_applied
                            ? `Enabled (Clip Limit ${qcResult.clahe_clip_limit}, ${qcResult.clahe_grid_size})`
                            : 'Disabled'}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                <div className="border border-slate-200 rounded-lg bg-white p-5 space-y-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h3 className="text-sm font-semibold text-slate-900">
                        Scale Calibration Matrix
                      </h3>
                      <p className="text-xs text-slate-500 mt-0.5">
                        Toggle calibrated millimeter-per-pixel photogrammetry or pixel-only fallback.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-1 p-1 bg-slate-100 border border-slate-200 rounded-md">
                    <button
                      type="button"
                      onClick={() => setQcIsCalibrated(true)}
                      className={`flex-1 px-2.5 py-1.5 text-xs font-medium rounded transition-colors cursor-pointer ${
                        qcIsCalibrated
                          ? 'bg-white text-slate-900 shadow-2xs'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      Calibrated Frame (mm/px)
                    </button>
                    <button
                      type="button"
                      onClick={() => setQcIsCalibrated(false)}
                      className={`flex-1 px-2.5 py-1.5 text-xs font-medium rounded transition-colors cursor-pointer ${
                        !qcIsCalibrated
                          ? 'bg-white text-slate-900 shadow-2xs'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      Uncalibrated Frame (px only)
                    </button>
                  </div>

                  {qcIsCalibrated ? (
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-xs text-slate-600 mb-1">
                          Fiducial Reference (mm)
                        </label>
                        <input
                          type="number"
                          min={10}
                          max={2000}
                          value={qcRefMm}
                          onChange={(e) => setQcRefMm(Number(e.target.value) || 100)}
                          className="w-full px-3 py-2 text-xs font-mono bg-slate-50 border border-slate-300 rounded-md text-slate-900 tabular-nums"
                        />
                      </div>
                      <div>
                        <label className="block text-xs text-slate-600 mb-1">
                          Measured Span (px)
                        </label>
                        <input
                          type="number"
                          min={10}
                          max={5000}
                          value={qcRefPx}
                          onChange={(e) => setQcRefPx(Number(e.target.value) || 294)}
                          className="w-full px-3 py-2 text-xs font-mono bg-slate-50 border border-slate-300 rounded-md text-slate-900 tabular-nums"
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="p-3 rounded-md bg-amber-50 border border-amber-200 text-xs text-amber-800">
                      No fiducial scale target detected. Crack length and aperture estimation will run
                      in calibrated pixel-space units (px) until a reference scale is supplied.
                    </div>
                  )}

                  <div className="pt-3 border-t border-slate-200 flex items-center justify-between text-xs">
                    <span className="text-slate-600">Computed Spatial Resolution:</span>
                    <span className="font-mono text-sm font-semibold text-blue-600 tabular-nums">
                      {!qcIsCalibrated
                        ? 'Uncalibrated (px space)'
                        : qcResult
                        ? `${qcResult.scale_calibration_mm_per_px} mm/px`
                        : `${(qcRefMm / Math.max(qcRefPx, 1)).toFixed(4)} mm/px`}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ===================================================================
            WORKSPACE VIEW 3: AI INSPECTION & QUANTITATIVE ANALYSIS WORKSPACE
        =================================================================== */}
        {activeWorkspace === 'inspection' && selectedAsset && (
          <div className="space-y-6">
            <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 pb-5 border-b border-slate-200">
              <div>
                <p className="text-xs font-medium text-blue-600 mb-1">
                  03. Multi-Stage Computer Vision Pipeline · Classification · YOLO · Segmentation · Grad-CAM XAI
                </p>
                <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">
                  {inspectionSubTab === 'viewport'
                    ? selectedAsset.asset_name
                    : 'Model Evaluation, Benchmarks & Training Workbench'}
                </h1>
                <p className="text-xs text-slate-600 mt-1">
                  {selectedAsset.asset_type} · {selectedAsset.material_type} ·{' '}
                  {selectedAsset.image.is_calibrated === false
                    ? 'Uncalibrated Pixel-Space Mode'
                    : `Calibrated Scale ${selectedAsset.image.scale_mm_per_px} mm/px`}
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-3 shrink-0">
                <div className="flex items-center gap-1 p-1 bg-slate-100 border border-slate-200 rounded-lg">
                  <button
                    type="button"
                    onClick={() => setInspectionSubTab('viewport')}
                    className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      inspectionSubTab === 'viewport'
                        ? 'bg-white text-slate-900 shadow-2xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Visual Inspection & Masks
                  </button>
                  <button
                    type="button"
                    onClick={() => setInspectionSubTab('benchmarks')}
                    className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      inspectionSubTab === 'benchmarks'
                        ? 'bg-white text-slate-900 shadow-2xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Model Benchmarks & Training
                  </button>
                </div>

                {inspectionSubTab === 'viewport' && (
                  <select
                    value={selectedAsset.asset_id}
                    onChange={(e) => setSelectedAssetId(Number(e.target.value))}
                    className="px-3 py-2 text-xs font-medium bg-white border border-slate-300 rounded-lg text-slate-900 focus:outline-none focus:border-blue-600"
                  >
                    {assets.map((a) => (
                      <option key={a.asset_id} value={a.asset_id}>
                        #{a.asset_id} — {a.asset_name} ({a.detection.defect_class})
                      </option>
                    ))}
                  </select>
                )}

                <button
                  type="button"
                  onClick={() => setActiveWorkspace('map')}
                  className="flex items-center gap-1 px-3.5 py-2 text-xs font-medium bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                >
                  <span>Next: Spatial Map</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {inspectionSubTab === 'viewport' ? (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                <div className="lg:col-span-7 space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-3 bg-white border border-slate-200 rounded-lg p-3">
                    <div className="flex items-center gap-1 p-1 bg-slate-100 border border-slate-200 rounded-md overflow-x-auto">
                      {(
                        [
                          { id: 'raw', label: 'Raw RGB' },
                          { id: 'yolo', label: 'YOLO Multi-Class Boxes' },
                          { id: 'unet', label: 'Crack Segmentation Mask' },
                          { id: 'gradcam', label: 'Grad-CAM XAI' },
                        ] as Array<{ id: ViewportLayer; label: string }>
                      ).map((tab) => (
                        <button
                          key={tab.id}
                          type="button"
                          onClick={() => setViewportLayer(tab.id)}
                          className={`px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                            viewportLayer === tab.id
                              ? 'bg-blue-600 text-white shadow-2xs'
                              : 'text-slate-600 hover:text-slate-900'
                          }`}
                        >
                          {tab.label}
                        </button>
                      ))}
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                      {viewportLayer !== 'raw' && (
                        <div className="flex items-center gap-2 text-xs">
                          <span className="text-slate-600 whitespace-nowrap">Opacity:</span>
                          <input
                            type="range"
                            min={20}
                            max={100}
                            value={overlayOpacity}
                            onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                            className="w-20 accent-blue-600"
                          />
                          <span className="font-mono text-slate-900 font-medium tabular-nums w-8 text-right">
                            {overlayOpacity}%
                          </span>
                        </div>
                      )}
                      <button
                        type="button"
                        onClick={() => handleDownloadMaskSvg(selectedAsset)}
                        className="px-2.5 py-1.5 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-800 rounded flex items-center gap-1.5 whitespace-nowrap cursor-pointer"
                      >
                        <Download className="w-3.5 h-3.5" />
                        <span>Export Mask SVG</span>
                      </button>
                    </div>
                  </div>

                  {/* Active Pipeline Model Selectors & Baseline Classifier Readout */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 bg-white border border-slate-200 rounded-lg p-3 text-xs">
                    <div className="flex flex-col justify-center">
                      <span className="text-[11px] text-slate-500">Baseline Binary Classifier</span>
                      <span className="font-mono font-semibold text-slate-900">
                        {selectedAsset.detection.baseline_classification?.label || 'Cracked'} (
                        {(
                          (selectedAsset.detection.baseline_classification?.confidence || 0.96) * 100
                        ).toFixed(1)}
                        %) · ResNet-50
                      </span>
                    </div>
                    <div>
                      <label className="block text-[11px] text-slate-500 mb-1">
                        Detection Engine
                      </label>
                      <select
                        value={activeDetectorModel}
                        onChange={(e) =>
                          setActiveDetectorModel(
                            e.target.value as 'YOLOv11x' | 'YOLOv8x' | 'RT-DETR-L'
                          )
                        }
                        className="w-full px-2.5 py-1 bg-slate-50 border border-slate-200 rounded text-slate-900 font-mono text-xs"
                      >
                        <option value="YOLOv11x">YOLOv11x (Selected · 94.6% mAP)</option>
                        <option value="YOLOv8x">YOLOv8x (92.8% mAP)</option>
                        <option value="RT-DETR-L">RT-DETR-L Transformer (93.9% mAP)</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-[11px] text-slate-500 mb-1">
                        Segmentation Engine
                      </label>
                      <select
                        value={activeSegModel}
                        onChange={(e) =>
                          setActiveSegModel(
                            e.target.value as 'SegFormer-B2' | 'Attention U-Net' | 'Swin-UNet'
                          )
                        }
                        className="w-full px-2.5 py-1 bg-slate-50 border border-slate-200 rounded text-slate-900 font-mono text-xs"
                      >
                        <option value="SegFormer-B2">SegFormer-B2 (Selected · 0.879 IoU)</option>
                        <option value="Attention U-Net">Attention U-Net (0.842 IoU)</option>
                        <option value="Swin-UNet">Swin-UNet (0.864 IoU)</option>
                      </select>
                    </div>
                  </div>

                  <div className="relative rounded-lg overflow-hidden border border-slate-200 bg-slate-900 aspect-[4/3]">
                    <ResilientInfrastructureImage
                      src={selectedAsset.image.storage_url}
                      alt={selectedAsset.asset_name}
                      className="w-full h-full object-cover"
                    />

                    <svg
                      viewBox="0 0 100 100"
                      preserveAspectRatio="none"
                      className="absolute inset-0 w-full h-full pointer-events-none"
                      style={{ opacity: overlayOpacity / 100 }}
                    >
                      <defs>
                        <radialGradient id="gradcamHotspotPrimaryLight" cx="50%" cy="50%" r="50%">
                          <stop offset="0%" stopColor="#EF4444" stopOpacity="0.88" />
                          <stop offset="45%" stopColor="#F59E0B" stopOpacity="0.55" />
                          <stop offset="80%" stopColor="#3B82F6" stopOpacity="0.25" />
                          <stop offset="100%" stopColor="#1E3A8A" stopOpacity="0" />
                        </radialGradient>
                      </defs>

                      {viewportLayer === 'yolo' && (
                        <g>
                          {(selectedAsset.detection.multi_class_boxes &&
                          selectedAsset.detection.multi_class_boxes.length > 0
                            ? selectedAsset.detection.multi_class_boxes
                            : [
                                {
                                  label: selectedAsset.detection.defect_class,
                                  confidence: selectedAsset.detection.confidence_score,
                                  xmin: selectedAsset.detection.bbox_xmin,
                                  ymin: selectedAsset.detection.bbox_ymin,
                                  xmax: selectedAsset.detection.bbox_xmax,
                                  ymax: selectedAsset.detection.bbox_ymax,
                                },
                              ]
                          ).map((box, idx) => {
                            const strokeHex =
                              idx === 0 ? '#38BDF8' : idx === 1 ? '#F59E0B' : '#A78BFA';
                            return (
                              <g key={idx}>
                                <rect
                                  x={box.xmin}
                                  y={box.ymin}
                                  width={box.xmax - box.xmin}
                                  height={box.ymax - box.ymin}
                                  fill={
                                    idx === 0
                                      ? 'rgba(37, 99, 235, 0.14)'
                                      : 'rgba(245, 158, 11, 0.12)'
                                  }
                                  stroke={strokeHex}
                                  strokeWidth="0.7"
                                  strokeDasharray={idx === 0 ? '2,1' : undefined}
                                />
                                <rect
                                  x={box.xmin}
                                  y={Math.max(box.ymin - 4.4, 0.5)}
                                  width={34}
                                  height={4.2}
                                  rx={0.6}
                                  fill="rgba(15, 23, 42, 0.88)"
                                />
                                <text
                                  x={box.xmin + 1.2}
                                  y={Math.max(box.ymin - 1.5, 3.2)}
                                  fill="#FFFFFF"
                                  fontSize="2.4"
                                  fontFamily="monospace"
                                >
                                  {box.label} {(box.confidence * 100).toFixed(0)}%
                                </text>
                              </g>
                            );
                          })}
                        </g>
                      )}

                      {viewportLayer === 'unet' && (
                        <g>
                          {selectedAsset.segmentation.svg_mask_paths.map((pathStr, idx) => (
                            <g key={idx}>
                              <path
                                d={pathStr}
                                fill="none"
                                stroke="#EF4444"
                                strokeWidth="3.2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                opacity="0.45"
                              />
                              <path
                                d={pathStr}
                                fill="none"
                                stroke="#38BDF8"
                                strokeWidth="0.9"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              />
                            </g>
                          ))}
                        </g>
                      )}

                      {viewportLayer === 'gradcam' && (
                        <g>
                          {selectedAsset.segmentation.gradcam_hotspots.map((spot, idx) => (
                            <g key={idx}>
                              <ellipse
                                cx={spot.cx}
                                cy={spot.cy}
                                rx={spot.rx}
                                ry={spot.ry}
                                fill="url(#gradcamHotspotPrimaryLight)"
                              />
                              <circle
                                cx={spot.cx}
                                cy={spot.cy}
                                r="1.2"
                                fill="#FFFFFF"
                                stroke="#DC2626"
                                strokeWidth="0.5"
                              />
                            </g>
                          ))}
                        </g>
                      )}
                    </svg>

                    <div className="absolute top-3 left-3 bg-black/80 border border-white/15 rounded px-3 py-2 text-xs font-mono text-white space-y-0.5">
                      <div>
                        {viewportLayer === 'raw' && 'MODE: RAW CALIBRATED RGB FRAME'}
                        {viewportLayer === 'yolo' &&
                          `${activeDetectorModel}: ${selectedAsset.detection.defect_class} (${(
                            selectedAsset.detection.confidence_score * 100
                          ).toFixed(1)}%) · ${
                            selectedAsset.detection.multi_class_boxes?.length || 1
                          } Detections`}
                        {viewportLayer === 'unet' &&
                          `${activeSegModel}: DENSITY ${selectedAsset.segmentation.crack_density_pct}% · IoU ${
                            selectedAsset.segmentation.segmentation_metrics?.iou ?? 0.86
                          }`}
                        {viewportLayer === 'gradcam' && 'GRAD-CAM XAI: LAYER4 CONV ATTRIBUTION'}
                      </div>
                      <div className="text-[11px] text-slate-300">
                        BBox: [{selectedAsset.detection.bbox_xmin}%,{' '}
                        {selectedAsset.detection.bbox_ymin}%, {selectedAsset.detection.bbox_xmax}%,{' '}
                        {selectedAsset.detection.bbox_ymax}%] · {selectedAsset.image.width_px}×
                        {selectedAsset.image.height_px}px
                      </div>
                    </div>

                    {viewportLayer === 'gradcam' && (
                      <div className="absolute bottom-3 left-3 right-3 bg-black/80 border border-white/15 rounded px-3 py-2 flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
                        {selectedAsset.segmentation.gradcam_hotspots.map((h, i) => (
                          <div key={i} className="text-white">
                            <span className="text-amber-300">
                              Activation {(h.weight * 100).toFixed(0)}%:
                            </span>{' '}
                            {h.label}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                <div className="lg:col-span-5 space-y-5">
                  <div className="border border-slate-200 rounded-lg bg-white p-5 space-y-4">
                    <div className="flex items-center justify-between pb-3 border-b border-slate-200">
                      <div>
                        <h3 className="text-sm font-semibold text-slate-900">
                          Quantitative Fracture & Geometry Telemetry
                        </h3>
                        <p className="text-xs text-slate-500">
                          {selectedAsset.image.is_calibrated === false
                            ? 'Uncalibrated pixel-space skeleton metrics'
                            : 'Derived from U-Net medial skeleton & photogrammetric scale factor'}
                        </p>
                      </div>
                      <span className="font-mono text-xs font-semibold text-slate-800 tabular-nums">
                        Health {selectedAsset.health_score}/100
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-3 font-mono tabular-nums">
                      <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                        <span className="block text-[11px] font-sans text-slate-500">
                          Total Crack Length
                        </span>
                        <span className="text-base font-semibold text-slate-900">
                          {selectedAsset.image.is_calibrated === false
                            ? `${
                                selectedAsset.segmentation.crack_geometry?.skeleton_length_px ??
                                Math.round(selectedAsset.segmentation.crack_length_mm * 2.8)
                              } px`
                            : `${selectedAsset.segmentation.crack_length_mm.toFixed(1)} mm`}
                        </span>
                      </div>
                      <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                        <span className="block text-[11px] font-sans text-slate-500">
                          Max Crack Width
                        </span>
                        <span className="text-base font-semibold text-amber-700">
                          {selectedAsset.image.is_calibrated === false
                            ? `${
                                selectedAsset.segmentation.crack_geometry?.max_aperture_px ?? 12.4
                              } px`
                            : `${selectedAsset.segmentation.max_width_mm.toFixed(2)} mm`}
                        </span>
                      </div>
                      <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                        <span className="block text-[11px] font-sans text-slate-500">
                          Mean Crack Width
                        </span>
                        <span className="text-base font-semibold text-slate-900">
                          {selectedAsset.image.is_calibrated === false
                            ? `${
                                selectedAsset.segmentation.crack_geometry?.mean_aperture_px ?? 6.2
                              } px`
                            : `${selectedAsset.segmentation.avg_width_mm.toFixed(2)} mm`}
                        </span>
                      </div>
                      <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                        <span className="block text-[11px] font-sans text-slate-500">
                          Surface Crack Density
                        </span>
                        <span className="text-base font-semibold text-blue-600">
                          {selectedAsset.segmentation.crack_density_pct.toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    {/* Crack Geometry Extraction */}
                    {selectedAsset.segmentation.crack_geometry && (
                      <div className="pt-3 border-t border-slate-200">
                        <div className="text-[11px] font-semibold text-slate-700 mb-2">
                          Crack Geometry Extraction (Medial Axis & Topology)
                        </div>
                        <div className="grid grid-cols-4 gap-2 text-center font-mono text-xs tabular-nums">
                          <div className="p-2 bg-slate-50 border border-slate-200 rounded">
                            <span className="block text-[10px] font-sans text-slate-500">
                              Tortuosity
                            </span>
                            <span className="font-semibold text-slate-900">
                              {selectedAsset.segmentation.crack_geometry.tortuosity_index}
                            </span>
                          </div>
                          <div className="p-2 bg-slate-50 border border-slate-200 rounded">
                            <span className="block text-[10px] font-sans text-slate-500">
                              Orientation
                            </span>
                            <span className="font-semibold text-slate-900">
                              {selectedAsset.segmentation.crack_geometry.dominant_orientation_deg}°
                            </span>
                          </div>
                          <div className="p-2 bg-slate-50 border border-slate-200 rounded">
                            <span className="block text-[10px] font-sans text-slate-500">
                              Branches
                            </span>
                            <span className="font-semibold text-slate-900">
                              {selectedAsset.segmentation.crack_geometry.branch_count}
                            </span>
                          </div>
                          <div className="p-2 bg-slate-50 border border-slate-200 rounded">
                            <span className="block text-[10px] font-sans text-slate-500">
                              Junctions
                            </span>
                            <span className="font-semibold text-slate-900">
                              {selectedAsset.segmentation.crack_geometry.junction_nodes}
                            </span>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Segmentation Evaluation Metrics */}
                    {selectedAsset.segmentation.segmentation_metrics && (
                      <div className="pt-3 border-t border-slate-200">
                        <div className="text-[11px] font-semibold text-slate-700 mb-2">
                          Segmentation Quality Metrics ({activeSegModel})
                        </div>
                        <div className="grid grid-cols-5 gap-1.5 text-center font-mono text-xs tabular-nums">
                          <div className="p-1.5 bg-slate-50 border border-slate-200 rounded">
                            <span className="block text-[10px] font-sans text-slate-500">IoU</span>
                            <span className="font-semibold text-emerald-700">
                              {selectedAsset.segmentation.segmentation_metrics.iou}
                            </span>
                          </div>
                          <div className="p-1.5 bg-slate-50 border border-slate-200 rounded">
                            <span className="block text-[10px] font-sans text-slate-500">Dice</span>
                            <span className="font-semibold text-slate-900">
                              {selectedAsset.segmentation.segmentation_metrics.dice}
                            </span>
                          </div>
                          <div className="p-1.5 bg-slate-50 border border-slate-200 rounded">
                            <span className="block text-[10px] font-sans text-slate-500">
                              Px Acc
                            </span>
                            <span className="font-semibold text-slate-900">
                              {selectedAsset.segmentation.segmentation_metrics.pixel_accuracy}%
                            </span>
                          </div>
                          <div className="p-1.5 bg-slate-50 border border-slate-200 rounded">
                            <span className="block text-[10px] font-sans text-slate-500">
                              B-F1
                            </span>
                            <span className="font-semibold text-slate-900">
                              {selectedAsset.segmentation.segmentation_metrics.boundary_f1}
                            </span>
                          </div>
                          <div className="p-1.5 bg-slate-50 border border-slate-200 rounded">
                            <span className="block text-[10px] font-sans text-slate-500">
                              Mask px
                            </span>
                            <span className="font-semibold text-slate-900">
                              {(
                                selectedAsset.segmentation.segmentation_metrics.mask_area_px / 1000
                              ).toFixed(0)}
                              k
                            </span>
                          </div>
                        </div>
                      </div>
                    )}

                    <div className="pt-2 text-xs text-slate-500 flex items-center justify-between">
                      <span>Verification Status: {selectedAsset.assessment.approval_status}</span>
                      <span>Reviewer: {selectedAsset.assessment.verified_by || 'Civil Engineer'}</span>
                    </div>
                  </div>

                  <div className="border border-slate-200 rounded-lg bg-white p-5 space-y-4">
                    <div className="flex items-center justify-between pb-3 border-b border-slate-200">
                      <div>
                        <h3 className="text-sm font-semibold text-slate-900">
                          Civil Engineer Override & Sign-Off
                        </h3>
                        <p className="text-xs font-mono text-slate-500">
                          POST /api/v1/assessments/{selectedAsset.assessment.assessment_id}/override
                        </p>
                      </div>
                      {!activePermissions.canOverrideAI && (
                        <span className="flex items-center gap-1 text-xs font-medium text-amber-700">
                          <Lock className="w-3.5 h-3.5" />
                          <span>Read-Only Role</span>
                        </span>
                      )}
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                      <div>
                        <label className="block text-slate-600 font-medium mb-1">
                          Defect Class
                        </label>
                        <select
                          disabled={!activePermissions.canOverrideAI}
                          value={overrideDefectClass}
                          onChange={(e) => setOverrideDefectClass(e.target.value as DefectClassType)}
                          className="w-full px-2.5 py-2 bg-slate-50 border border-slate-300 rounded text-slate-900 disabled:opacity-50"
                        >
                          <option value="D00 Longitudinal">D00 Longitudinal</option>
                          <option value="D10 Transverse">D10 Transverse</option>
                          <option value="D20 Alligator">D20 Alligator</option>
                          <option value="D40 Pothole">D40 Pothole</option>
                          <option value="Spalling">Spalling</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-slate-600 font-medium mb-1">
                          Severity Level
                        </label>
                        <select
                          disabled={!activePermissions.canOverrideAI}
                          value={overrideSeverity}
                          onChange={(e) =>
                            setOverrideSeverity(e.target.value as SeverityLevelType)
                          }
                          className="w-full px-2.5 py-2 bg-slate-50 border border-slate-300 rounded text-slate-900 disabled:opacity-50"
                        >
                          <option value="Minor">Minor</option>
                          <option value="Moderate">Moderate</option>
                          <option value="Severe">Severe</option>
                          <option value="Critical">Critical</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-slate-600 font-medium mb-1">Priority</label>
                        <select
                          disabled={!activePermissions.canOverrideAI}
                          value={overridePriority}
                          onChange={(e) => setOverridePriority(e.target.value as PriorityType)}
                          className="w-full px-2.5 py-2 bg-slate-50 border border-slate-300 rounded text-slate-900 disabled:opacity-50"
                        >
                          <option value="Routine">Routine</option>
                          <option value="Scheduled">Scheduled</option>
                          <option value="Urgent">Urgent</option>
                          <option value="Immediate">Immediate</option>
                        </select>
                      </div>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-600 mb-1">
                        Structural Engineering Assessment & Remediation Directive
                      </label>
                      <textarea
                        rows={3}
                        disabled={!activePermissions.canOverrideAI}
                        value={overrideComments}
                        onChange={(e) => setOverrideComments(e.target.value)}
                        className="w-full p-3 text-xs bg-slate-50 border border-slate-300 rounded text-slate-900 focus:outline-none focus:border-blue-600 focus:bg-white disabled:opacity-50"
                      />
                    </div>

                    <div className="flex flex-wrap items-center gap-2 pt-2">
                      <button
                        type="button"
                        disabled={isSavingOverride}
                        onClick={() => handleEngineeringOverride('Approved')}
                        className="px-3.5 py-2 text-xs font-semibold bg-emerald-600 hover:bg-emerald-700 text-white rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                      >
                        Approve Assessment
                      </button>
                      <button
                        type="button"
                        disabled={isSavingOverride}
                        onClick={() => handleEngineeringOverride('Reclassified')}
                        className="px-3.5 py-2 text-xs font-semibold bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                      >
                        Reclassify & Update
                      </button>
                      <button
                        type="button"
                        disabled={isSavingOverride}
                        onClick={() => {
                          setOverrideSeverity('Critical');
                          setOverridePriority('Immediate');
                          handleEngineeringOverride('Escalated');
                        }}
                        className="px-3.5 py-2 text-xs font-semibold bg-red-600 hover:bg-red-700 text-white rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                      >
                        Escalate Immediate
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              /* MODEL BENCHMARKS, TRAINING & EVALUATION REPORT SUB-TAB */
              benchmarksData && (
                <div className="space-y-6">
                  {/* Best Selected Summary Callouts */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="bg-white border border-slate-200 rounded-lg p-4">
                      <span className="text-[11px] font-medium text-slate-500">
                        Best Baseline Classifier
                      </span>
                      <div className="text-sm font-semibold text-slate-900 mt-1">
                        {benchmarksData.summary.best_baseline_classifier}
                      </div>
                      <span className="text-xs font-mono text-emerald-700">
                        SDNET2018 Binary Crack Triage
                      </span>
                    </div>
                    <div className="bg-white border border-slate-200 rounded-lg p-4">
                      <span className="text-[11px] font-medium text-slate-500">
                        Best Multi-Class Detector
                      </span>
                      <div className="text-sm font-semibold text-slate-900 mt-1">
                        {benchmarksData.summary.best_detector}
                      </div>
                      <span className="text-xs font-mono text-blue-600">
                        RDD2022 + Spalling Detection
                      </span>
                    </div>
                    <div className="bg-white border border-slate-200 rounded-lg p-4">
                      <span className="text-[11px] font-medium text-slate-500">
                        Best Baseline Segmentation
                      </span>
                      <div className="text-sm font-semibold text-slate-900 mt-1">
                        {benchmarksData.summary.best_baseline_segmentation}
                      </div>
                      <span className="text-xs font-mono text-slate-700">
                        Crack500 + DeepCrack Masks
                      </span>
                    </div>
                    <div className="bg-white border border-slate-200 rounded-lg p-4">
                      <span className="text-[11px] font-medium text-slate-500">
                        Best Advanced Transformer
                      </span>
                      <div className="text-sm font-semibold text-slate-900 mt-1">
                        {benchmarksData.summary.best_advanced_transformer}
                      </div>
                      <span className="text-xs font-mono text-emerald-700">
                        Selected Production Backbone
                      </span>
                    </div>
                  </div>

                  {/* Interactive Epoch Training & Evaluation Report Bar */}
                  <form
                    onSubmit={handleTrainModel}
                    className="bg-white border border-slate-200 rounded-lg p-4 flex flex-wrap items-center justify-between gap-4"
                  >
                    <div className="flex flex-wrap items-center gap-3 text-xs">
                      <span className="font-semibold text-slate-900 flex items-center gap-1.5">
                        <Play className="w-3.5 h-3.5 text-blue-600" />
                        <span>Run Training / Evaluation Pass:</span>
                      </span>
                      <select
                        value={trainModelId}
                        onChange={(e) => setTrainModelId(e.target.value)}
                        className="px-3 py-1.5 bg-slate-50 border border-slate-300 rounded text-slate-900 font-medium"
                      >
                        {benchmarksData.models.map((m) => (
                          <option key={m.model_id} value={m.model_id}>
                            {m.name} ({m.family})
                          </option>
                        ))}
                      </select>
                      <select
                        value={trainEpochs}
                        onChange={(e) => setTrainEpochs(Number(e.target.value))}
                        className="px-2.5 py-1.5 bg-slate-50 border border-slate-300 rounded text-slate-900 font-mono"
                      >
                        <option value={5}>+5 Epochs</option>
                        <option value={10}>+10 Epochs</option>
                        <option value={25}>+25 Epochs</option>
                      </select>
                      <button
                        type="submit"
                        disabled={isTrainingModel}
                        className="px-3.5 py-1.5 text-xs font-semibold bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors cursor-pointer disabled:opacity-50"
                      >
                        {isTrainingModel ? 'Evaluating...' : 'Train & Evaluate'}
                      </button>
                    </div>

                    <button
                      type="button"
                      onClick={handleExportModelReport}
                      className="px-3.5 py-1.5 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-900 rounded flex items-center gap-1.5 cursor-pointer"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>Export Model Eval Report (JSON)</span>
                    </button>
                  </form>

                  {/* 3 Grouped Model Comparison Tables */}
                  {(
                    [
                      'Baseline Classification',
                      'Object Detection',
                      'Segmentation & Advanced CNN/Transformer',
                    ] as const
                  ).map((family) => {
                    const familyModels = benchmarksData.models.filter((m) => m.family === family);
                    return (
                      <div
                        key={family}
                        className="bg-white border border-slate-200 rounded-lg overflow-hidden"
                      >
                        <div className="px-5 py-3 border-b border-slate-200 bg-slate-50/70 flex items-center justify-between">
                          <h3 className="text-sm font-semibold text-slate-900">{family}</h3>
                          <span className="text-xs font-mono text-slate-500">
                            {familyModels.length} Evaluated Architectures
                          </span>
                        </div>
                        <div className="overflow-x-auto">
                          <table className="w-full text-left border-collapse text-xs">
                            <thead>
                              <tr className="border-b border-slate-200 text-[11px] font-semibold text-slate-500">
                                <th className="py-2.5 px-4">Model Architecture</th>
                                <th className="py-2.5 px-3">Dataset</th>
                                <th className="py-2.5 px-3 text-right">Primary Metric</th>
                                <th className="py-2.5 px-3 text-right">Secondary</th>
                                <th className="py-2.5 px-3 text-right">Prec / Rec</th>
                                <th className="py-2.5 px-3 text-right">F1 / Dice</th>
                                <th className="py-2.5 px-3 text-right">Latency</th>
                                <th className="py-2.5 px-3 text-right">Epochs</th>
                                <th className="py-2.5 px-4 text-right">Selection</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-200/70 font-mono tabular-nums">
                              {familyModels.map((m) => (
                                <tr key={m.model_id} className="hover:bg-slate-50">
                                  <td className="py-3 px-4 font-sans">
                                    <div className="font-semibold text-slate-900">{m.name}</div>
                                    <div className="text-[11px] text-slate-500">
                                      {m.architecture} · {m.params_m}M params
                                    </div>
                                  </td>
                                  <td className="py-3 px-3 font-sans text-slate-600">{m.dataset}</td>
                                  <td className="py-3 px-3 text-right font-semibold text-blue-600">
                                    {m.primary_metric_value} ({m.primary_metric_name})
                                  </td>
                                  <td className="py-3 px-3 text-right text-slate-800">
                                    {m.secondary_metric_value} ({m.secondary_metric_name})
                                  </td>
                                  <td className="py-3 px-3 text-right text-slate-700">
                                    {m.precision}% / {m.recall}%
                                  </td>
                                  <td className="py-3 px-3 text-right text-slate-900">
                                    {m.f1_or_dice}%
                                  </td>
                                  <td className="py-3 px-3 text-right text-slate-700">
                                    {m.latency_ms} ms
                                  </td>
                                  <td className="py-3 px-3 text-right text-slate-700">
                                    {m.epochs_trained}
                                  </td>
                                  <td className="py-3 px-4 text-right font-sans">
                                    {m.is_selected ? (
                                      <span className="text-xs font-semibold text-emerald-700">
                                        Selected Active
                                      </span>
                                    ) : (
                                      <button
                                        type="button"
                                        onClick={() => handleSelectModel(m.model_id)}
                                        className="px-2.5 py-1 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-800 rounded cursor-pointer"
                                      >
                                        Select Model
                                      </button>
                                    )}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    );
                  })}

                  {/* Epoch Convergence SVG Chart & Per-Class AP Matrix */}
                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                    <div className="lg:col-span-6 bg-white border border-slate-200 rounded-lg p-5 space-y-3">
                      <div className="flex items-center justify-between">
                        <h3 className="text-sm font-semibold text-slate-900">
                          Training Epoch Convergence (IoU & mAP)
                        </h3>
                        <span className="text-[11px] font-mono text-slate-500">Epochs 1 → 80</span>
                      </div>
                      <svg viewBox="0 0 400 180" className="w-full h-44 bg-slate-50 rounded border border-slate-200">
                        {[0.5, 0.7, 0.9].map((v) => {
                          const y = 160 - (v - 0.4) * 240;
                          return (
                            <g key={v}>
                              <line x1="36" y1={y} x2="386" y2={y} stroke="#E2E8F0" strokeDasharray="3,3" />
                              <text x="6" y={y + 3} fontSize="9" fill="#64748B" fontFamily="monospace">
                                {(v * 100).toFixed(0)}%
                              </text>
                            </g>
                          );
                        })}
                        <polyline
                          fill="none"
                          stroke="#2563EB"
                          strokeWidth="2.2"
                          points={benchmarksData.epoch_curve
                            .map(
                              (pt, i) =>
                                `${40 + i * 48},${160 - (pt.segformer_iou - 0.4) * 240}`
                            )
                            .join(' ')}
                        />
                        <polyline
                          fill="none"
                          stroke="#10B981"
                          strokeWidth="2"
                          points={benchmarksData.epoch_curve
                            .map((pt, i) => `${40 + i * 48},${160 - (pt.yolo_map - 0.4) * 240}`)
                            .join(' ')}
                        />
                        <polyline
                          fill="none"
                          stroke="#F59E0B"
                          strokeWidth="1.8"
                          strokeDasharray="4,2"
                          points={benchmarksData.epoch_curve
                            .map((pt, i) => `${40 + i * 48},${160 - (pt.unet_iou - 0.4) * 240}`)
                            .join(' ')}
                        />
                      </svg>
                      <div className="flex flex-wrap items-center gap-4 text-[11px] font-mono text-slate-600">
                        <span className="text-blue-600 font-semibold">— SegFormer-B2 IoU</span>
                        <span className="text-emerald-700 font-semibold">— YOLOv11x mAP@50</span>
                        <span className="text-amber-700 font-semibold">-- Attention U-Net IoU</span>
                      </div>
                    </div>

                    <div className="lg:col-span-6 bg-white border border-slate-200 rounded-lg p-5 space-y-3">
                      <h3 className="text-sm font-semibold text-slate-900">
                        Multi-Class Surface Defect Performance Breakdown
                      </h3>
                      <div className="overflow-x-auto">
                        <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
                          <thead>
                            <tr className="border-b border-slate-200 text-[11px] font-sans font-semibold text-slate-500">
                              <th className="py-2 pr-3">Defect Class</th>
                              <th className="py-2 px-2 text-right">YOLOv11 AP</th>
                              <th className="py-2 px-2 text-right">SegFormer IoU</th>
                              <th className="py-2 px-2 text-right">U-Net IoU</th>
                              <th className="py-2 pl-2 text-right">Samples</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-200/70">
                            {benchmarksData.per_class_ap.map((row) => (
                              <tr key={row.defect_class}>
                                <td className="py-2.5 pr-3 font-sans font-medium text-slate-900">
                                  {row.defect_class}
                                </td>
                                <td className="py-2.5 px-2 text-right text-blue-600 font-semibold">
                                  {row.ap_yolov11}%
                                </td>
                                <td className="py-2.5 px-2 text-right text-emerald-700 font-semibold">
                                  {row.iou_segformer}
                                </td>
                                <td className="py-2.5 px-2 text-right text-slate-700">
                                  {row.iou_unet}
                                </td>
                                <td className="py-2.5 pl-2 text-right text-slate-500">
                                  {row.samples}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                </div>
              )
            )}
          </div>
        )}

        {/* ===================================================================
            WORKSPACE VIEW 4: GEOSPATIAL DAMAGE MAPPING DASHBOARD
        =================================================================== */}
        {activeWorkspace === 'map' && (
          <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 pb-5 border-b border-slate-200">
              <div>
                <p className="text-xs font-medium text-blue-600 mb-1">
                  04. PostGIS Spatial Intelligence · Leaflet Vector Pin & Defect Density Layer
                </p>
                <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">
                  Geospatial Damage Mapping Dashboard
                </h1>
                <p className="text-sm text-slate-600 mt-1 max-w-2xl">
                  Spatial distribution of civil infrastructure defects color-coded and labeled by
                  engineering severity level (Minor, Moderate, Severe, Critical).
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-3 shrink-0">
                <div className="flex items-center gap-1 p-1 bg-slate-100 border border-slate-200 rounded-lg">
                  {(['All', 'Critical', 'Severe', 'Moderate', 'Minor'] as const).map((lvl) => (
                    <button
                      key={lvl}
                      type="button"
                      onClick={() => setMapSeverityFilter(lvl)}
                      className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                        mapSeverityFilter === lvl
                          ? 'bg-white text-slate-900 shadow-2xs'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      {lvl}
                    </button>
                  ))}
                </div>

                <button
                  type="button"
                  onClick={() => setShowHeatmapLayer((v) => !v)}
                  className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    showHeatmapLayer
                      ? 'bg-amber-50 border-amber-300 text-amber-800'
                      : 'bg-white border-slate-300 text-slate-600'
                  }`}
                >
                  Defect Density Radius: {showHeatmapLayer ? 'ON' : 'OFF'}
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
              <div className="lg:col-span-8 border border-slate-200 rounded-lg overflow-hidden bg-slate-100 min-h-[520px] relative">
                <div ref={mapContainerRef} className="w-full h-[520px] z-10" />
              </div>

              <div className="lg:col-span-4 border border-slate-200 rounded-lg bg-white p-5 flex flex-col justify-between space-y-6">
                {selectedAsset ? (
                  <div className="space-y-4">
                    <div className="pb-3 border-b border-slate-200">
                      <span className="text-xs font-mono text-slate-500 tabular-nums">
                        PostGIS Point({selectedAsset.longitude.toFixed(4)}{' '}
                        {selectedAsset.latitude.toFixed(4)})
                      </span>
                      <h3 className="text-base font-semibold text-slate-900 mt-0.5">
                        {selectedAsset.asset_name}
                      </h3>
                      <p className="text-xs text-slate-500">
                        {selectedAsset.asset_type} · {selectedAsset.responsible_department}
                      </p>
                    </div>

                    <div className="relative h-40 rounded overflow-hidden border border-slate-200">
                      <ResilientInfrastructureImage
                        src={selectedAsset.image.storage_url}
                        alt={selectedAsset.asset_name}
                        className="w-full h-full object-cover"
                      />
                    </div>

                    <div className="space-y-2 text-xs font-mono tabular-nums">
                      <div className="flex justify-between py-1.5 border-b border-slate-100">
                        <span className="font-sans text-slate-500">Primary Defect Class</span>
                        <span className="text-slate-900 font-semibold">
                          {selectedAsset.detection.defect_class}
                        </span>
                      </div>
                      <div className="flex justify-between py-1.5 border-b border-slate-100">
                        <span className="font-sans text-slate-500">Severity & Priority</span>
                        <span
                          className={`font-semibold ${
                            getSeverityMeta(selectedAsset.assessment.severity_level).textClass
                          }`}
                        >
                          {selectedAsset.assessment.severity_level} ·{' '}
                          {selectedAsset.assessment.maintenance_priority}
                        </span>
                      </div>
                      <div className="flex justify-between py-1.5 border-b border-slate-100">
                        <span className="font-sans text-slate-500">Crack Length / Max Width</span>
                        <span className="text-slate-900">
                          {selectedAsset.segmentation.crack_length_mm} mm /{' '}
                          {selectedAsset.segmentation.max_width_mm} mm
                        </span>
                      </div>
                      <div className="flex justify-between py-1.5">
                        <span className="font-sans text-slate-500">Surface Crack Density</span>
                        <span className="text-slate-900 font-semibold">
                          {selectedAsset.segmentation.crack_density_pct}%
                        </span>
                      </div>
                    </div>

                    <p className="text-xs text-slate-600 leading-relaxed">
                      {selectedAsset.assessment.engineer_comments}
                    </p>
                  </div>
                ) : null}

                <div className="pt-4 border-t border-slate-200 flex items-center justify-between gap-3">
                  <button
                    type="button"
                    onClick={() => {
                      if (leafletMapRef.current && selectedAsset) {
                        leafletMapRef.current.flyTo(
                          [selectedAsset.latitude, selectedAsset.longitude],
                          13
                        );
                      }
                    }}
                    className="px-3 py-2 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    Center Pin on Map
                  </button>
                  <button
                    type="button"
                    onClick={() => setActiveWorkspace('inspection')}
                    className="px-4 py-2 text-xs font-semibold bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    Inspect Defect Masks
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ===================================================================
            WORKSPACE VIEW 5: REPORT EXPORT & COMPLIANCE SUMMARY
        =================================================================== */}
        {activeWorkspace === 'reports' && (
          <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 pb-5 border-b border-slate-200">
              <div>
                <p className="text-xs font-medium text-blue-600 mb-1">
                  05. Damage Analytics, Deliverable Verification & PE Compliance Export
                </p>
                <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">
                  {reportsSubTab === 'analytics'
                    ? 'Damage Analytics & Tasks Verification Matrix'
                    : 'Report Export & Compliance Summary'}
                </h1>
                <p className="text-sm text-slate-600 mt-1 max-w-2xl">
                  Portfolio-wide fracture telemetry, multi-class damage distribution, verification of
                  all 21 Tasks & 17 Expected Outcomes, and auditable PE sign-off dossiers.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-3 shrink-0">
                <div className="flex items-center gap-1 p-1 bg-slate-100 border border-slate-200 rounded-lg">
                  <button
                    type="button"
                    onClick={() => setReportsSubTab('analytics')}
                    className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      reportsSubTab === 'analytics'
                        ? 'bg-white text-slate-900 shadow-2xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Damage Analytics & Tasks
                  </button>
                  <button
                    type="button"
                    onClick={() => setReportsSubTab('compliance')}
                    className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      reportsSubTab === 'compliance'
                        ? 'bg-white text-slate-900 shadow-2xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Compliance Reports
                  </button>
                </div>

                <button
                  type="button"
                  onClick={handleDownloadCsv}
                  className="px-3.5 py-2 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-900 rounded-lg transition-colors whitespace-nowrap shrink-0 flex items-center gap-1.5 cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>CSV Ledger</span>
                </button>
                <button
                  type="button"
                  onClick={() => window.print()}
                  className="px-3.5 py-2 text-xs font-semibold bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors whitespace-nowrap shrink-0 flex items-center gap-1.5 cursor-pointer"
                >
                  <Printer className="w-3.5 h-3.5" />
                  <span>Print PDF</span>
                </button>
              </div>
            </div>

            {reportsSubTab === 'analytics' ? (
              <div className="space-y-6">
                {/* Portfolio Damage Analytics KPI Strip */}
                {analyticsSummary && (
                  <>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                      <div className="bg-white border border-slate-200 rounded-lg p-4">
                        <span className="text-xs text-slate-500">Total Mapped Crack Length</span>
                        <div className="text-xl font-semibold font-mono text-slate-900 tabular-nums mt-1">
                          {analyticsSummary.total_crack_length_mm.toLocaleString()} mm
                        </div>
                        <span className="text-[11px] font-mono text-slate-500">
                          Mean Density {analyticsSummary.mean_crack_density_pct}%
                        </span>
                      </div>
                      <div className="bg-white border border-slate-200 rounded-lg p-4">
                        <span className="text-xs text-slate-500">Peak Observed Aperture</span>
                        <div className="text-xl font-semibold font-mono text-amber-700 tabular-nums mt-1">
                          {analyticsSummary.max_aperture_mm.toFixed(2)} mm
                        </div>
                        <span className="text-[11px] font-mono text-slate-500">
                          ACI 224R Critical Envelope
                        </span>
                      </div>
                      <div className="bg-white border border-slate-200 rounded-lg p-4">
                        <span className="text-xs text-slate-500">Calibrated Photogrammetry</span>
                        <div className="text-xl font-semibold font-mono text-emerald-700 tabular-nums mt-1">
                          {analyticsSummary.calibrated_assets} / {analyticsSummary.total_assets} Assets
                        </div>
                        <span className="text-[11px] font-mono text-slate-500">
                          Fiducial mm/px Scale Active
                        </span>
                      </div>
                      <div className="bg-white border border-slate-200 rounded-lg p-4">
                        <span className="text-xs text-slate-500">Estimated Remediation Budget</span>
                        <div className="text-xl font-semibold font-mono text-blue-600 tabular-nums mt-1">
                          ${analyticsSummary.total_remediation_budget_usd.toLocaleString()}
                        </div>
                        <span className="text-[11px] font-mono text-slate-500">
                          Prioritized Work Orders
                        </span>
                      </div>
                    </div>

                    {/* Severity Distribution, Multi-Class Defect Breakdown & Asset Type Risk */}
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                      <div className="lg:col-span-4 bg-white border border-slate-200 rounded-lg p-5 space-y-4">
                        <h3 className="text-sm font-semibold text-slate-900">
                          Severity Level Distribution
                        </h3>
                        <div className="space-y-3 text-xs">
                          {analyticsSummary.severity_breakdown.map((item) => {
                            const barColor =
                              item.level === 'Critical'
                                ? 'bg-red-600'
                                : item.level === 'Severe'
                                ? 'bg-amber-600'
                                : item.level === 'Moderate'
                                ? 'bg-yellow-500'
                                : 'bg-emerald-600';
                            return (
                              <div key={item.level} className="space-y-1">
                                <div className="flex items-center justify-between font-mono tabular-nums">
                                  <span className="font-sans font-medium text-slate-800">
                                    {item.level}
                                  </span>
                                  <span className="text-slate-600">
                                    {item.count} assets ({item.pct}%)
                                  </span>
                                </div>
                                <div className="h-2 w-full bg-slate-100 rounded overflow-hidden">
                                  <div
                                    className={`h-full ${barColor}`}
                                    style={{ width: `${Math.max(item.pct, 4)}%` }}
                                  />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>

                      <div className="lg:col-span-4 bg-white border border-slate-200 rounded-lg p-5 space-y-3">
                        <h3 className="text-sm font-semibold text-slate-900">
                          Multi-Class Defect Analytics
                        </h3>
                        <div className="overflow-x-auto">
                          <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
                            <thead>
                              <tr className="border-b border-slate-200 text-[11px] font-sans text-slate-500">
                                <th className="py-2 pr-2">Class</th>
                                <th className="py-2 px-2 text-right">Count</th>
                                <th className="py-2 px-2 text-right">Avg Width</th>
                                <th className="py-2 pl-2 text-right">Density</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-200/70">
                              {analyticsSummary.defect_class_breakdown.map((d) => (
                                <tr key={d.defect_class}>
                                  <td className="py-2 pr-2 font-sans font-medium text-slate-900">
                                    {d.defect_class}
                                  </td>
                                  <td className="py-2 px-2 text-right text-slate-700">{d.count}</td>
                                  <td className="py-2 px-2 text-right text-slate-800">
                                    {d.avg_max_width_mm} mm
                                  </td>
                                  <td className="py-2 pl-2 text-right text-blue-600">
                                    {d.avg_density_pct}%
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>

                      <div className="lg:col-span-4 bg-white border border-slate-200 rounded-lg p-5 space-y-3">
                        <h3 className="text-sm font-semibold text-slate-900">
                          Asset Type Structural Risk
                        </h3>
                        <div className="overflow-x-auto">
                          <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
                            <thead>
                              <tr className="border-b border-slate-200 text-[11px] font-sans text-slate-500">
                                <th className="py-2 pr-2">Structure</th>
                                <th className="py-2 px-2 text-right">Health</th>
                                <th className="py-2 pl-2 text-right">Total Length</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-200/70">
                              {analyticsSummary.asset_type_risk.map((r) => (
                                <tr key={r.asset_type}>
                                  <td className="py-2 pr-2 font-sans font-medium text-slate-900">
                                    {r.asset_type} ({r.count})
                                  </td>
                                  <td className="py-2 px-2 text-right text-slate-800">
                                    {r.avg_health}/100
                                  </td>
                                  <td className="py-2 pl-2 text-right text-slate-700">
                                    {r.total_crack_length_mm} mm
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </div>
                  </>
                )}

                {/* 21 Tasks & 17 Expected Outcomes Verification Matrix */}
                {tasksStatus && (
                  <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
                    <div className="p-5 border-b border-slate-200 flex flex-col lg:flex-row lg:items-center justify-between gap-4 bg-slate-50/70">
                      <div>
                        <h2 className="text-base font-semibold text-slate-900">
                          Program Scope: {tasksStatus.tasks_completed}/{tasksStatus.tasks_total} Tasks Complete ·{' '}
                          {tasksStatus.outcomes_operational}/{tasksStatus.outcomes_total} Expected Outcomes Operational
                        </h2>
                        <p className="text-xs text-slate-500 mt-0.5">
                          Live end-to-end verification matrix linking each engineering task and expected outcome to its active module and API endpoint.
                        </p>
                      </div>

                      <div className="flex flex-wrap items-center gap-2">
                        <div className="flex items-center gap-1 p-1 bg-white border border-slate-200 rounded-md">
                          {(['All', 'Image & CV', 'Models & XAI', 'GIS & Maintenance'] as const).map(
                            (cat) => (
                              <button
                                key={cat}
                                type="button"
                                onClick={() => setTaskCategoryFilter(cat)}
                                className={`px-2.5 py-1 text-xs font-medium rounded transition-colors cursor-pointer ${
                                  taskCategoryFilter === cat
                                    ? 'bg-blue-600 text-white'
                                    : 'text-slate-600 hover:text-slate-900'
                                }`}
                              >
                                {cat}
                              </button>
                            )
                          )}
                        </div>
                        <input
                          type="text"
                          value={taskSearchQuery}
                          onChange={(e) => setTaskSearchQuery(e.target.value)}
                          placeholder="Filter tasks..."
                          className="px-3 py-1.5 text-xs bg-white border border-slate-300 rounded-md text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-blue-600"
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-12 divide-y lg:divide-y-0 lg:divide-x divide-slate-200">
                      {/* 21 Tasks Ledger */}
                      <div className="lg:col-span-7 p-5 space-y-3">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                          21 Engineering & AI Pipeline Tasks
                        </h3>
                        <div className="divide-y divide-slate-200/70 max-h-[540px] overflow-y-auto pr-1">
                          {tasksStatus.tasks
                            .filter(
                              (t) =>
                                (taskCategoryFilter === 'All' ||
                                  t.category === taskCategoryFilter) &&
                                (!taskSearchQuery.trim() ||
                                  t.title.toLowerCase().includes(taskSearchQuery.toLowerCase()) ||
                                  t.evidence.toLowerCase().includes(taskSearchQuery.toLowerCase()))
                            )
                            .map((t) => (
                              <div
                                key={t.id}
                                className="py-3 flex items-start justify-between gap-3 text-xs"
                              >
                                <div className="space-y-0.5">
                                  <div className="flex items-center gap-2">
                                    <span className="font-mono font-semibold text-blue-600">
                                      {t.id}
                                    </span>
                                    <span className="font-semibold text-slate-900">{t.title}</span>
                                    <span className="text-[11px] font-medium text-emerald-700">
                                      · {t.status}
                                    </span>
                                  </div>
                                  <div className="text-slate-600">{t.evidence}</div>
                                  <div className="text-[11px] font-mono text-slate-500">
                                    {t.category} · {t.endpoint}
                                  </div>
                                </div>
                                <button
                                  type="button"
                                  onClick={() => {
                                    setActiveWorkspace(t.workspace);
                                    if (t.id === 'T10' || t.id === 'T11' || t.id === 'T13') {
                                      setInspectionSubTab('benchmarks');
                                    } else if (t.workspace === 'inspection') {
                                      setInspectionSubTab('viewport');
                                    } else if (t.id === 'T19') {
                                      setAssetLayout('priority');
                                    } else if (t.id === 'T18' && selectedAsset) {
                                      handleOpenHistory(selectedAsset.asset_id);
                                    }
                                  }}
                                  className="px-2.5 py-1 text-[11px] font-medium bg-slate-100 hover:bg-slate-200 text-slate-800 rounded whitespace-nowrap shrink-0 cursor-pointer"
                                >
                                  Open Module
                                </button>
                              </div>
                            ))}
                        </div>
                      </div>

                      {/* 17 Expected Outcomes Ledger */}
                      <div className="lg:col-span-5 p-5 space-y-3 bg-slate-50/40">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                          17 Expected Outcomes
                        </h3>
                        <div className="divide-y divide-slate-200/70 max-h-[540px] overflow-y-auto pr-1">
                          {tasksStatus.outcomes.map((o) => (
                            <div
                              key={o.id}
                              className="py-2.5 flex items-center justify-between gap-3 text-xs"
                            >
                              <div>
                                <div className="font-medium text-slate-900">
                                  <span className="font-mono text-slate-500 mr-2">{o.id}</span>
                                  {o.title}
                                </div>
                                <div className="text-[11px] font-mono text-slate-500">
                                  Linked: {o.linked_tasks.join(', ')}
                                </div>
                              </div>
                              <span className="flex items-center gap-1 text-[11px] font-semibold text-emerald-700 whitespace-nowrap shrink-0">
                                <CheckCircle2 className="w-3.5 h-3.5" />
                                <span>{o.status}</span>
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                <div className="lg:col-span-4 border border-slate-200 rounded-lg bg-white p-5 space-y-5">
                  <h3 className="text-sm font-semibold text-slate-900 pb-3 border-b border-slate-200 flex items-center gap-2">
                    <FileCheck2 className="w-4 h-4 text-blue-600" />
                    <span>Compliance Export Configuration</span>
                  </h3>

                  <div className="space-y-4 text-xs">
                    <div>
                      <label className="block font-medium text-slate-700 mb-1.5">
                        Included Asset Scope
                      </label>
                      <select
                        value={reportScope}
                        onChange={(e) =>
                          setReportScope(e.target.value as 'all' | 'critical_severe')
                        }
                        className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-md text-slate-900"
                      >
                        <option value="all">
                          Full Civil Infrastructure Inventory ({assets.length})
                        </option>
                        <option value="critical_severe">
                          Critical & Severe Remediation Queue Only ({portfolioStats.criticalOrSevere})
                        </option>
                      </select>
                    </div>

                    <div className="space-y-2 pt-2">
                      <label className="flex items-center gap-2 text-slate-700 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={includeXaiAppendix}
                          onChange={(e) => setIncludeXaiAppendix(e.target.checked)}
                          className="accent-blue-600"
                        />
                        <span>Include Grad-CAM XAI Attribution Coordinates</span>
                      </label>
                      <label className="flex items-center gap-2 text-slate-700 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={includePostgisCoords}
                          onChange={(e) => setIncludePostgisCoords(e.target.checked)}
                          className="accent-blue-600"
                        />
                        <span>Include PostGIS EPSG:4326 Spatial Coordinates</span>
                      </label>
                    </div>

                    <div className="pt-4 border-t border-slate-200 space-y-3">
                      <h4 className="font-semibold text-slate-900">
                        Certifying Civil Engineer Sign-Off
                      </h4>
                      <div>
                        <label className="block text-slate-600 mb-1">Engineer Full Name</label>
                        <input
                          type="text"
                          value={engineerSignOffName}
                          onChange={(e) => setEngineerSignOffName(e.target.value)}
                          className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-md text-slate-900"
                        />
                      </div>
                      <div>
                        <label className="block text-slate-600 mb-1">PE License / Stamp ID</label>
                        <input
                          type="text"
                          value={engineerLicenseNo}
                          onChange={(e) => setEngineerLicenseNo(e.target.value)}
                          className="w-full px-3 py-2 font-mono bg-slate-50 border border-slate-300 rounded-md text-slate-900"
                        />
                      </div>
                      <button
                        type="button"
                        onClick={handleExportModelReport}
                        className="w-full px-3 py-2 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-900 rounded-md flex items-center justify-center gap-1.5 cursor-pointer"
                      >
                        <Download className="w-3.5 h-3.5" />
                        <span>Export Model Evaluation Report (JSON)</span>
                      </button>
                    </div>
                  </div>
                </div>

                <div className="lg:col-span-8 border border-slate-200 rounded-lg bg-white p-6 space-y-6">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-slate-200">
                    <div>
                      <p className="text-xs font-mono text-slate-500">
                        DOCUMENT REF: INFRASCAN-ACI224R-2026
                      </p>
                      <h2 className="text-lg font-semibold text-slate-900 mt-0.5">
                        Civil Infrastructure Structural Condition & Defect Ledger
                      </h2>
                    </div>
                    <div className="text-xs font-mono text-slate-500 sm:text-right tabular-nums">
                      <div>Generated: {new Date().toISOString().slice(0, 10)}</div>
                      <div>
                        Certifying PE: {engineerSignOffName || currentUser.full_name} (
                        {engineerLicenseNo})
                      </div>
                    </div>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-slate-200 text-[11px] font-semibold text-slate-500">
                          <th className="py-2.5 pr-4">Asset</th>
                          <th className="py-2.5 px-3">Defect Class</th>
                          <th className="py-2.5 px-3 text-right">Length (mm)</th>
                          <th className="py-2.5 px-3 text-right">Max Width (mm)</th>
                          <th className="py-2.5 px-3 text-right">Density</th>
                          <th className="py-2.5 pl-3">Severity · Priority</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200/70">
                        {(reportScope === 'critical_severe'
                          ? assets.filter(
                              (a) =>
                                a.assessment.severity_level === 'Critical' ||
                                a.assessment.severity_level === 'Severe'
                            )
                          : assets
                        ).map((a) => (
                          <tr key={a.asset_id}>
                            <td className="py-3 pr-4">
                              <div className="font-semibold text-slate-900">{a.asset_name}</div>
                              {includePostgisCoords && (
                                <div className="text-[11px] font-mono text-slate-500 tabular-nums">
                                  {a.latitude.toFixed(4)}°N, {a.longitude.toFixed(4)}°W
                                </div>
                              )}
                            </td>
                            <td className="py-3 px-3 font-mono text-slate-800">
                              {a.detection.defect_class}
                            </td>
                            <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-800">
                              {a.segmentation.crack_length_mm.toFixed(1)}
                            </td>
                            <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-800">
                              {a.segmentation.max_width_mm.toFixed(2)}
                            </td>
                            <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-800">
                              {a.segmentation.crack_density_pct.toFixed(1)}%
                            </td>
                            <td className="py-3 pl-3">
                              <span
                                className={`font-semibold ${
                                  getSeverityMeta(a.assessment.severity_level).textClass
                                }`}
                              >
                                {a.assessment.severity_level} · {a.assessment.maintenance_priority}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="pt-5 border-t border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-4 text-xs text-slate-500">
                    <div>
                      Attested in accordance with ACI 224R-01 & FHWA Bridge Inspector Reference
                      Manual. All frames passed Laplacian sharpness threshold (Var(ΔI) &gt;= 100.0).
                    </div>
                    <div className="font-mono font-semibold text-slate-800 shrink-0">
                      Signed: {engineerSignOffName || currentUser.full_name} · {engineerLicenseNo}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* =====================================================================
          MODAL: ADMINISTRATOR OVERRIDE REQUIRED TO CHANGE LOCKED ACTIVE ROLE
      ===================================================================== */}
      {pendingOverrideRole && (
        <div className="fixed inset-0 z-50 bg-slate-900/45 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="w-full max-w-md bg-white border border-slate-200 rounded-xl p-6 space-y-5 shadow-xl">
            <div className="flex items-start justify-between gap-3 pb-3 border-b border-slate-200">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-amber-50 border border-amber-200 flex items-center justify-center text-amber-700 shrink-0">
                  <Unlock className="w-4 h-4" />
                </div>
                <div>
                  <h2 className="text-base font-semibold text-slate-900">
                    Administrator Role Override Required
                  </h2>
                  <p className="text-xs text-slate-500">
                    Active role is locked to <span className="font-semibold">{currentUser.role}</span>
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setPendingOverrideRole(null)}
                className="text-xs text-slate-500 hover:text-slate-900"
              >
                Cancel
              </button>
            </div>

            <form onSubmit={handleConfirmAdminRoleOverride} className="space-y-4 text-xs">
              <p className="text-slate-600 leading-relaxed">
                Session roles are locked at Sign-In. To override the active session role from{' '}
                <span className="font-semibold text-slate-900">{currentUser.role}</span> to{' '}
                <span className="font-semibold text-blue-600">{pendingOverrideRole}</span>, enter
                the Administrator Access Key.
              </p>

              <div>
                <label className="block font-medium text-slate-700 mb-1.5">
                  Administrator Access Key
                </label>
                <input
                  type="password"
                  required
                  autoFocus
                  placeholder="Enter Administrator Access Key"
                  value={adminOverrideKeyInput}
                  onChange={(e) => setAdminOverrideKeyInput(e.target.value)}
                  className="w-full px-3 py-2 font-mono bg-slate-50 border border-slate-300 rounded-lg text-slate-900 focus:outline-none focus:border-blue-600 focus:bg-white"
                />
              </div>

              {adminOverrideError && (
                <div className="p-2.5 rounded-lg bg-red-50 border border-red-200 text-red-700 font-medium">
                  {adminOverrideError}
                </div>
              )}

              <div className="pt-2 flex items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setPendingOverrideRole(null)}
                  className="px-3.5 py-2 text-xs font-medium text-slate-600 hover:text-slate-900"
                >
                  Keep Current Role
                </button>
                <button
                  type="submit"
                  disabled={isOverridingRole}
                  className="px-4 py-2 text-xs font-semibold bg-blue-600 hover:bg-blue-700 text-white rounded-lg cursor-pointer"
                >
                  {isOverridingRole ? 'Verifying...' : `Authorize Override to ${pendingOverrideRole}`}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* =====================================================================
          SLIDE-OVER DRAWER 1: ASSET INSPECTION HISTORY (GET /api/v1/assets/{id}/history)
      ===================================================================== */}
      {historyDrawerAsset && (
        <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-xs flex justify-end">
          <div className="w-full max-w-xl bg-white border-l border-slate-200 h-full flex flex-col justify-between overflow-y-auto p-6 shadow-xl">
            <div className="space-y-6">
              <div className="flex items-start justify-between pb-4 border-b border-slate-200">
                <div>
                  <span className="text-xs font-mono text-slate-500">
                    GET /api/v1/assets/{historyDrawerAsset.asset.asset_id}/history
                  </span>
                  <h2 className="text-lg font-semibold text-slate-900 mt-0.5">
                    {historyDrawerAsset.asset.asset_name} — Inspection Timeline
                  </h2>
                  <p className="text-xs text-slate-500">
                    Longitudinal fracture progression & historical U-Net aperture measurements
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setHistoryDrawerAsset(null)}
                  className="text-xs font-medium text-slate-500 hover:text-slate-900"
                >
                  Close
                </button>
              </div>

              {/* Side-by-Side Historical Inspection Comparison */}
              {historyDrawerAsset.history.length >= 2 && (() => {
                const hist = historyDrawerAsset.history;
                const entryA = hist[Math.min(compareIdxA, hist.length - 1)];
                const entryB = hist[Math.min(compareIdxB, hist.length - 1)];
                const widthDelta = Number((entryB.max_width_mm - entryA.max_width_mm).toFixed(2));
                const densityDelta = Number(
                  (entryB.crack_density_pct - entryA.crack_density_pct).toFixed(1)
                );
                const maxW = Math.max(...hist.map((h) => h.max_width_mm), 1);

                return (
                  <div className="border border-slate-200 rounded-lg bg-slate-50 p-4 space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-xs font-semibold text-slate-900 flex items-center gap-1.5">
                        <GitCompare className="w-3.5 h-3.5 text-blue-600" />
                        <span>Historical Inspection Comparison</span>
                      </h3>
                      <span className="text-[11px] font-mono text-slate-500">
                        Aperture Progression
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div>
                        <label className="block text-[11px] text-slate-500 mb-1">
                          Baseline Survey (T0)
                        </label>
                        <select
                          value={compareIdxA}
                          onChange={(e) => setCompareIdxA(Number(e.target.value))}
                          className="w-full px-2.5 py-1.5 bg-white border border-slate-300 rounded text-slate-900 font-mono"
                        >
                          {hist.map((h, i) => (
                            <option key={h.inspection_id} value={i}>
                              #{h.inspection_id} ({new Date(h.inspection_date).toISOString().slice(0, 10)})
                            </option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label className="block text-[11px] text-slate-500 mb-1">
                          Follow-Up Survey (T1)
                        </label>
                        <select
                          value={compareIdxB}
                          onChange={(e) => setCompareIdxB(Number(e.target.value))}
                          className="w-full px-2.5 py-1.5 bg-white border border-slate-300 rounded text-slate-900 font-mono"
                        >
                          {hist.map((h, i) => (
                            <option key={h.inspection_id} value={i}>
                              #{h.inspection_id} ({new Date(h.inspection_date).toISOString().slice(0, 10)})
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-2 text-xs font-mono tabular-nums">
                      <div className="p-2.5 bg-white border border-slate-200 rounded">
                        <span className="block text-[10px] font-sans text-slate-500">
                          Δ Max Width
                        </span>
                        <span
                          className={`font-semibold ${
                            widthDelta > 0 ? 'text-red-600' : 'text-emerald-700'
                          }`}
                        >
                          {widthDelta > 0 ? `+${widthDelta}` : widthDelta} mm
                        </span>
                      </div>
                      <div className="p-2.5 bg-white border border-slate-200 rounded">
                        <span className="block text-[10px] font-sans text-slate-500">
                          Δ Surface Density
                        </span>
                        <span
                          className={`font-semibold ${
                            densityDelta > 0 ? 'text-amber-700' : 'text-emerald-700'
                          }`}
                        >
                          {densityDelta > 0 ? `+${densityDelta}` : densityDelta}%
                        </span>
                      </div>
                      <div className="p-2.5 bg-white border border-slate-200 rounded">
                        <span className="block text-[10px] font-sans text-slate-500">
                          Severity Shift
                        </span>
                        <span className="font-sans font-semibold text-slate-900">
                          {entryA.severity_level} → {entryB.severity_level}
                        </span>
                      </div>
                    </div>

                    {/* Inline Progression Sparkline */}
                    <div className="bg-white border border-slate-200 rounded p-2.5">
                      <div className="flex items-end gap-3 h-14">
                        {[...hist].reverse().map((h) => {
                          const pct = Math.max(Math.round((h.max_width_mm / maxW) * 100), 18);
                          return (
                            <div
                              key={h.inspection_id}
                              className="flex-1 flex flex-col items-center justify-end h-full gap-1"
                            >
                              <span className="text-[10px] font-mono text-slate-600 tabular-nums">
                                {h.max_width_mm}mm
                              </span>
                              <div
                                className="w-full bg-blue-600/80 rounded-t"
                                style={{ height: `${pct}%` }}
                              />
                              <span className="text-[10px] font-mono text-slate-500">
                                {new Date(h.inspection_date).toISOString().slice(0, 7)}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                );
              })()}

              <div className="space-y-4">
                {historyDrawerAsset.history.map((item) => {
                  const sev = getSeverityMeta(item.severity_level);
                  return (
                    <div
                      key={item.inspection_id}
                      className="border border-slate-200 rounded-lg bg-slate-50 p-4 space-y-2"
                    >
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-mono font-medium text-slate-700 tabular-nums">
                          Inspection #{item.inspection_id} ·{' '}
                          {new Date(item.inspection_date).toISOString().slice(0, 10)}
                        </span>
                        <span className={`font-semibold ${sev.textClass}`}>
                          {item.severity_level} · {item.status}
                        </span>
                      </div>
                      <div className="text-xs font-mono text-slate-900 tabular-nums">
                        {item.defect_class} · Max Width: {item.max_width_mm} mm · Crack Density:{' '}
                        {item.crack_density_pct}%
                      </div>
                      <p className="text-xs text-slate-600">{item.notes}</p>
                      <div className="text-[11px] text-slate-500">Unit: {item.inspector}</div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="pt-4 border-t border-slate-200 flex justify-end">
              <button
                type="button"
                onClick={() => setHistoryDrawerAsset(null)}
                className="px-4 py-2 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-md"
              >
                Close History
              </button>
            </div>
          </div>
        </div>
      )}

      {/* =====================================================================
          MODAL: REGISTER NEW INFRASTRUCTURE ASSET (POST /api/v1/assets)
      ===================================================================== */}
      {showNewAssetModal && (
        <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="w-full max-w-lg bg-white border border-slate-200 rounded-xl p-6 space-y-5 shadow-xl">
            <div className="flex items-center justify-between pb-3 border-b border-slate-200">
              <div>
                <h2 className="text-base font-semibold text-slate-900">
                  Register Civil Infrastructure Asset
                </h2>
                <p className="text-xs text-slate-500">
                  Endpoint: POST /api/v1/assets (Authenticated as {currentUser.role})
                </p>
              </div>
              <button
                type="button"
                onClick={() => setShowNewAssetModal(false)}
                className="text-xs text-slate-500 hover:text-slate-900"
              >
                Cancel
              </button>
            </div>

            <form onSubmit={handleCreateAsset} className="space-y-4 text-xs">
              <div>
                <label className="block font-medium text-slate-700 mb-1">
                  Infrastructure Asset Name
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g., Bayshore Viaduct Pier 19 East Cap"
                  value={newAssetForm.asset_name}
                  onChange={(e) =>
                    setNewAssetForm({ ...newAssetForm, asset_name: e.target.value })
                  }
                  className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-md text-slate-900"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block font-medium text-slate-700 mb-1">Structure Type</label>
                  <select
                    value={newAssetForm.asset_type}
                    onChange={(e) =>
                      setNewAssetForm({ ...newAssetForm, asset_type: e.target.value })
                    }
                    className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-md text-slate-900"
                  >
                    <option value="Bridge">Bridge</option>
                    <option value="Pavement">Pavement</option>
                    <option value="Vertical Building">Vertical Building</option>
                    <option value="Tunnel">Tunnel</option>
                    <option value="Hydraulic Dam">Hydraulic Dam</option>
                  </select>
                </div>

                <div>
                  <label className="block font-medium text-slate-700 mb-1">
                    Construction Year
                  </label>
                  <input
                    type="number"
                    required
                    value={newAssetForm.construction_year}
                    onChange={(e) =>
                      setNewAssetForm({
                        ...newAssetForm,
                        construction_year: Number(e.target.value),
                      })
                    }
                    className="w-full px-3 py-2 font-mono bg-slate-50 border border-slate-300 rounded-md text-slate-900"
                  />
                </div>
              </div>

              <div>
                <label className="block font-medium text-slate-700 mb-1">
                  Structural Material Specification
                </label>
                <input
                  type="text"
                  required
                  value={newAssetForm.material_type}
                  onChange={(e) =>
                    setNewAssetForm({ ...newAssetForm, material_type: e.target.value })
                  }
                  className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-md text-slate-900"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block font-medium text-slate-700 mb-1">
                    Latitude (EPSG:4326)
                  </label>
                  <input
                    type="number"
                    step="0.0001"
                    required
                    value={newAssetForm.latitude}
                    onChange={(e) =>
                      setNewAssetForm({ ...newAssetForm, latitude: Number(e.target.value) })
                    }
                    className="w-full px-3 py-2 font-mono bg-slate-50 border border-slate-300 rounded-md text-slate-900"
                  />
                </div>
                <div>
                  <label className="block font-medium text-slate-700 mb-1">
                    Longitude (EPSG:4326)
                  </label>
                  <input
                    type="number"
                    step="0.0001"
                    required
                    value={newAssetForm.longitude}
                    onChange={(e) =>
                      setNewAssetForm({ ...newAssetForm, longitude: Number(e.target.value) })
                    }
                    className="w-full px-3 py-2 font-mono bg-slate-50 border border-slate-300 rounded-md text-slate-900"
                  />
                </div>
              </div>

              <div>
                <label className="block font-medium text-slate-700 mb-1">
                  Responsible Engineering Department
                </label>
                <input
                  type="text"
                  required
                  value={newAssetForm.responsible_department}
                  onChange={(e) =>
                    setNewAssetForm({
                      ...newAssetForm,
                      responsible_department: e.target.value,
                    })
                  }
                  className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-md text-slate-900"
                />
              </div>

              <div className="pt-3 border-t border-slate-200 flex items-center justify-end gap-3">
                <button
                  type="button"
                  onClick={() => setShowNewAssetModal(false)}
                  className="px-4 py-2 text-xs text-slate-600 hover:text-slate-900"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 text-xs font-semibold bg-blue-600 hover:bg-blue-700 text-white rounded-md"
                >
                  Commit to Asset Registry
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* =====================================================================
          SLIDE-OVER DRAWER 2: PYTHON FASTAPI & SQLALCHEMY SOURCE CODE VIEWER
      ===================================================================== */}
      {showBackendDrawer && (
        <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-xs flex justify-end">
          <div className="w-full max-w-3xl bg-white border-l border-slate-200 h-full flex flex-col justify-between p-6 overflow-y-auto shadow-xl">
            <div className="space-y-4">
              <div className="flex items-start justify-between pb-4 border-b border-slate-200">
                <div>
                  <span className="text-xs font-mono text-blue-600">backend/main.py</span>
                  <h2 className="text-lg font-semibold text-slate-900 mt-0.5">
                    FastAPI + SQLAlchemy ORM + OAuth2 RBAC Backend
                  </h2>
                  <p className="text-xs text-slate-500">
                    Complete Python FastAPI backend source code and local uvicorn setup commands.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      navigator.clipboard.writeText(fastApiSource);
                      setCopiedCode(true);
                      setTimeout(() => setCopiedCode(false), 2000);
                    }}
                    className="px-3 py-1.5 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-md flex items-center gap-1.5"
                  >
                    {copiedCode ? (
                      <Check className="w-3.5 h-3.5 text-emerald-600" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                    <span>{copiedCode ? 'Copied' : 'Copy Python Code'}</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowBackendDrawer(false)}
                    className="px-3 py-1.5 text-xs font-medium text-slate-500 hover:text-slate-900"
                  >
                    Close
                  </button>
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 font-mono text-xs text-slate-200 space-y-1">
                <div className="text-slate-400"># Run Standalone Python FastAPI Backend:</div>
                <div>pip install -r backend/requirements.txt</div>
                <div>uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload</div>
              </div>

              <pre className="bg-slate-900 border border-slate-800 rounded-lg p-4 font-mono text-[11px] text-slate-200 overflow-x-auto max-h-[60vh]">
                <code>{fastApiSource}</code>
              </pre>
            </div>
          </div>
        </div>
      )}

      {/* Minimalist Quiet Footer */}
      <footer className="border-t border-slate-200 bg-white py-4 px-6 text-xs text-slate-500 mt-auto">
        <div className="max-w-[1440px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <div>
            InfraScan Civil Infrastructure Assessment Platform · YOLOv11 · U-Net · Grad-CAM XAI ·
            PostGIS Relational Schema
          </div>
          <div className="font-mono text-slate-600">
            Session Operator: {currentUser.full_name} · Locked Role: {currentUser.role}
          </div>
        </div>
      </footer>
    </div>
  );
}
