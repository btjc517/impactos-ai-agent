# Render + Supabase Setup Guide for ImpactOS AI

This guide walks you through setting up your ImpactOS AI system on Render with Supabase as the database.

## 🎯 What You'll Set Up on Render

**Choose "Web Services"** - This is your FastAPI application that handles queries from your web portal.

## 📋 Prerequisites

1. **GitHub Repository** - Your `impactos-ai-agent` repo (✅ already have this)
2. **Render Account** - Sign up at [render.com](https://render.com)
3. **Supabase Account** - Sign up at [supabase.com](https://supabase.com)
4. **OpenAI API Key** - For GPT-4 functionality

## 🗄️ Step 1: Set Up Supabase Database

### 1.1 Create Supabase Project
1. Go to [supabase.com](https://supabase.com) and sign in
2. Click "New Project"
3. Choose your organization
4. Fill in project details:
   - **Name**: `impactos-ai-db` (or any name you prefer)
   - **Database Password**: Create a strong password (save this!)
   - **Region**: Choose closest to your users
5. Click "Create new project"
6. Wait for project to be ready (2-3 minutes)

### 1.2 Get Database Connection String
1. In your Supabase project dashboard, go to **Settings** → **Database**
2. Scroll down to **Connection string** → **URI**
3. Copy the connection string (looks like):
   ```
   postgresql://postgres:[YOUR-PASSWORD]@db.abc123.supabase.co:5432/postgres
   ```
4. Replace `[YOUR-PASSWORD]` with your actual database password
5. **Save this connection string** - you'll need it for Render

### 1.3 Configure Database Security (Optional but Recommended)
1. Go to **Settings** → **API**
2. Note down your **Project URL** and **anon/public key** (for future use)
3. For now, we'll use the direct database connection

## 🚀 Step 2: Deploy to Render

### 2.1 Create Web Service
1. Go to [render.com](https://render.com) and sign in
2. Click **"New +"** → **"Web Service"**
3. Choose **"Build and deploy from a Git repository"**
4. Click **"Connect"** next to GitHub
5. Select your **`impactos-ai-agent`** repository
6. Click **"Connect"**

### 2.2 Configure Web Service
Fill in the service configuration:

**Basic Settings:**
- **Name**: `impactos-ai-api` (or your preferred name)
- **Region**: Choose closest to your users
- **Branch**: `develop` (or `main` if you've merged)
- **Root Directory**: Leave blank (uses repository root)

**Build & Deploy:**
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
 - **Start Command**: `python src/web_api.py --host 0.0.0.0 --port $PORT`

**Instance Type:**
- **Free**: Start with free tier (good for testing)
- **Starter**: $7/month (recommended for production)

### 2.3 Set Environment Variables
Click **"Advanced"** → **"Add Environment Variable"** and add:

| Key | Value |
|-----|-------|
| `OPENAI_API_KEY` | `sk-your-actual-openai-api-key` |
| `DATABASE_URL` | `postgresql://postgres:[password]@db.abc123.supabase.co:5432/postgres` |
| `PORT` | `8000` (usually auto-set by Render) |

**Important**: Replace the `DATABASE_URL` with your actual Supabase connection string from Step 1.2!

### 2.4 Deploy
1. Click **"Create Web Service"**
2. Render will start building and deploying your application
3. This takes 5-10 minutes for the first deployment
4. Watch the logs for any errors

## ✅ Step 3: Verify Deployment

### 3.1 Check Service Health
Once deployment completes:
1. Render will provide a URL like: `https://impactos-ai-api.onrender.com`
2. Test the health endpoint: `https://your-url.onrender.com/health`
3. You should see:
   ```json
   {
     "status": "healthy",
     "version": "1.0.0",
     "database_connected": true,
     "openai_configured": true
   }
   ```

### 3.2 Test API Functionality
Test the query endpoint:
```bash
curl -X POST "https://your-url.onrender.com/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "Hello, is the system working?"}'
```

### 3.3 View Interactive Documentation
Visit: `https://your-url.onrender.com/docs`
This shows all available API endpoints with testing interface.

## 🔧 Step 4: Configure Your Web Portal

Update your web portal to use the new production API:

```javascript
// Replace localhost with your Render URL
const API_BASE_URL = 'https://impactos-ai-api.onrender.com';

// Your existing query function will now work globally
const queryData = async (question) => {
    const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, show_accuracy: true })
    });
    return await response.json();
};
```

## 📊 Step 5: Data Migration (If You Have Existing Data)

If you have data in your local SQLite database, you'll need to migrate it:

### Option A: Upload Files via API
Use the `/upload` endpoint to re-process your data files:
```javascript
const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData
    });
    return await response.json();
};
```

### Option B: Direct Database Migration (Advanced)
1. Export data from SQLite
2. Import into Supabase using their dashboard or SQL commands

## 🎛️ Step 6: Configure Production Settings

### 6.1 Update CORS for Production
In your `src/web_api.py`, update CORS settings:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-web-portal-domain.com",  # Your actual website
        "https://localhost:3000",              # For local development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### 6.2 Enable Auto-Deploy
1. In Render dashboard, go to your service
2. Under **Settings** → **Build & Deploy**
3. Enable **"Auto-Deploy"** from your GitHub branch
4. Now every time you push to GitHub, Render will automatically redeploy

## 🔍 Monitoring and Maintenance

### Health Monitoring
Set up regular health checks in your web portal:
```javascript
const monitorAPI = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const health = await response.json();
        
        if (health.status !== 'healthy') {
            console.error('API health issue:', health);
            // Alert your team or show user message
        }
    } catch (error) {
        console.error('API unavailable:', error);
    }
};

// Check every 5 minutes
setInterval(monitorAPI, 5 * 60 * 1000);
```

### Viewing Logs
- Go to your Render service dashboard
- Click **"Logs"** to see real-time application logs
- Monitor for errors, performance issues, or usage patterns

## 💰 Cost Breakdown

### Render Costs:
- **Free Tier**: Good for testing, has limitations
- **Starter ($7/month)**: Recommended for production
  - Always-on service
  - Custom domains
  - More resources

### Supabase Costs:
- **Free Tier**: 500MB database, 2GB bandwidth/month
- **Pro ($25/month)**: 8GB database, 250GB bandwidth/month
- **Most small to medium applications stay in free tier**

### Total Monthly Cost:
- **Development/Testing**: $0 (both free tiers)
- **Small Production**: $7 (Render Starter + Supabase Free)
- **Medium Production**: $32 (Render Starter + Supabase Pro)

## 🚨 Troubleshooting

### Common Issues:

1. **Build Fails**
   - Check Python dependencies in `requirements.txt`
   - Verify build logs in Render dashboard
   - Ensure all files are committed to GitHub

2. **Database Connection Errors**
   - Verify `DATABASE_URL` environment variable
   - Check Supabase project is running
   - Ensure password is correct in connection string

3. **API Returns 500 Errors**
   - Check application logs in Render
   - Verify `OPENAI_API_KEY` is set correctly
   - Test database connection from Supabase dashboard

4. **CORS Errors from Web Portal**
   - Update `allow_origins` in `web_api.py`
   - Redeploy after making changes
   - Ensure your web portal uses HTTPS

### Getting Help:
- **Render Support**: [render.com/docs](https://render.com/docs)
- **Supabase Support**: [supabase.com/docs](https://supabase.com/docs)
- **Check service logs** for specific error messages

## ✅ Success Checklist

- [ ] Supabase project created and database connection string saved
- [ ] Render web service deployed successfully
- [ ] Environment variables configured (`OPENAI_API_KEY`, `DATABASE_URL`)
- [ ] Health endpoint returns "healthy" status
- [ ] API docs accessible at `/docs` endpoint
- [ ] Web portal updated to use production API URL
- [ ] Test query works end-to-end
- [ ] Auto-deploy enabled for future updates

Your ImpactOS AI system is now running in the cloud and accessible 24/7! 🎉 