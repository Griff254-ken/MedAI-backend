import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import mongoose from "mongoose";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import helmet from "helmet";
import morgan from "morgan";
import rateLimit from "express-rate-limit";
import compression from "compression";
import axios from "axios";
import multer from "multer";
import FormData from "form-data";
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

// Load environment variables
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const config = {
  PORT: process.env.PORT || 10000,
  NODE_ENV: process.env.NODE_ENV || "development",
  MONGODB_URI: process.env.MONGODB_URI,
  JWT_SECRET: process.env.JWT_SECRET || "secure_dev_secret",
  JWT_EXPIRATION: "7d",
  AI_PORT: process.env.AI_PORT || 8000,
  AI_HOST: "127.0.0.1",
  CORS_ORIGIN: process.env.CORS_ORIGIN || "*",
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
  ALLOWED_FILE_TYPES: ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
};

// ====== 1. Start Python AI Backend (Optimized for Render) ====== //
let pythonProcess = null;

const startPythonAI = () => {
  try {
    // Check if main.py exists before trying to start
    if (!fs.existsSync(path.join(__dirname, "main.py"))) {
      console.error("❌ main.py not found. AI backend cannot start.");
      return;
    }

    // On Render, we use the system 'python3' directly
    const pythonCmd = config.NODE_ENV === "production" ? "python3" : "python";
    
    console.log(`🤖 Initializing Python AI with command: ${pythonCmd}`);

    pythonProcess = spawn(pythonCmd, [
      "-m", "uvicorn", 
      "main:app", 
      "--host", config.AI_HOST,
      "--port", config.AI_PORT.toString(),
      "--log-level", "info"
    ], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONUNBUFFERED: "1" }
    });

    pythonProcess.stdout.on("data", (data) => console.log(`🤖 [AI]: ${data.toString().trim()}`));
    pythonProcess.stderr.on("data", (data) => console.error(`🤖 [AI-Error]: ${data.toString().trim()}`));
    
    pythonProcess.on("exit", (code) => {
      console.warn(`🤖 AI process exited with code ${code}. Restarting in 5s...`);
      setTimeout(startPythonAI, 5000); // Auto-restart if it crashes
    });

  } catch (error) {
    console.error("❌ Critical error starting Python AI:", error.message);
  }
};

// ====== 2. Database ====== //
mongoose.set('strictQuery', false);

const UserSchema = new mongoose.Schema({
  email: { 
    type: String, 
    unique: true, 
    required: true, 
    lowercase: true,
    trim: true
  },
  passwordHash: { type: String, required: true },
  name: { type: String, required: true },
  role: { type: String, enum: ["user", "doctor", "admin"], default: "user" },
  createdAt: { type: Date, default: Date.now }
});

// REMOVED: UserSchema.index({ email: 1 }); -> Fixed Duplicate Index Warning
UserSchema.index({ createdAt: -1 });

const User = mongoose.model("User", UserSchema);

const connectDB = async () => {
  try {
    await mongoose.connect(config.MONGODB_URI);
    console.log("✅ MongoDB connected successfully");
  } catch (err) {
    console.error("❌ MongoDB Connection Error:", err.message);
    if (config.NODE_ENV === "production") process.exit(1);
  }
};

// ====== 3. Express App ====== //
const app = express();

app.use(helmet({ crossOriginResourcePolicy: { policy: "cross-origin" } }));
app.use(cors({ origin: config.CORS_ORIGIN, credentials: true }));
app.use(compression());
app.use(express.json({ limit: "10mb" }));
app.use(morgan("dev"));

const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: config.MAX_FILE_SIZE }
});

// Middleware: Auth
const authenticateToken = (req, res, next) => {
  const token = req.headers["authorization"]?.split(" ")[1];
  if (!token) return res.status(401).json({ message: "Access denied" });

  jwt.verify(token, config.JWT_SECRET, (err, user) => {
    if (err) return res.status(403).json({ message: "Invalid token" });
    req.user = user;
    next();
  });
};

// ====== 4. Routes ====== //

app.get("/health", (req, res) => {
  res.json({
    status: "online",
    database: mongoose.connection.readyState === 1 ? "connected" : "error",
    ai_service: pythonProcess && !pythonProcess.killed ? "running" : "offline"
  });
});

app.post("/auth/register", async (req, res) => {
  try {
    const { email, password, name, role } = req.body;
    const passwordHash = await bcrypt.hash(password, 12);
    const user = new User({ email, passwordHash, name, role });
    await user.save();
    
    const token = jwt.sign({ id: user._id, email: user.email, role: user.role }, config.JWT_SECRET);
    res.status(201).json({ token, user: { id: user._id, name: user.name, email: user.email } });
  } catch (err) {
    res.status(400).json({ message: "Registration failed", error: err.message });
  }
});

app.post("/auth/login", async (req, res) => {
  const { email, password } = req.body;
  const user = await User.findOne({ email });
  if (user && await bcrypt.compare(password, user.passwordHash)) {
    const token = jwt.sign({ id: user._id, email: user.email, role: user.role }, config.JWT_SECRET);
    return res.json({ token, user: { id: user._id, name: user.name } });
  }
  res.status(401).json({ message: "Invalid credentials" });
});

app.post("/diagnostics/process", authenticateToken, upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ message: "No image provided" });

    const formData = new FormData();
    formData.append('file', req.file.buffer, { filename: 'image.jpg', contentType: req.file.mimetype });

    const aiServiceUrl = `http://${config.AI_HOST}:${config.AI_PORT}/diagnostics/process`;
    
    const response = await axios.post(aiServiceUrl, formData, {
      headers: { ...formData.getHeaders() },
      params: { type: req.body.type || "xray" },
      timeout: 30000 
    });

    res.json(response.data);
  } catch (error) {
    console.error("AI Bridge Error:", error.message);
    res.status(503).json({ message: "AI Service Unavailable", detail: error.message });
  }
});

// ====== 5. Execution ====== //
const start = async () => {
  await connectDB();
  startPythonAI();
  app.listen(config.PORT, () => {
    console.log(`🚀 MedAI Gateway Live on port ${config.PORT}`);
  });
};

start();
