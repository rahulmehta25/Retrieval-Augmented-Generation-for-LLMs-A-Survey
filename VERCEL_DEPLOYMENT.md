# Vercel Deployment Guide for RAG Frontend

This guide provides comprehensive instructions for deploying the RAG Knowledge System frontend to Vercel with optimal configuration for production use.

## Quick Start

1. **Connect to Vercel**
   - Install Vercel CLI: `npm install -g vercel`
   - Login: `vercel login`
   - Link project: `vercel link`

2. **Deploy**
   ```bash
   vercel --prod
   ```

## Configuration Files Overview

### Root Configuration: `/vercel.json`
- **Purpose**: Main deployment configuration for the entire repository
- **Key Features**:
  - Framework detection (Vite)
  - Build command routing to frontend directory
  - API proxy configuration for backend integration
  - Security headers and CORS settings
  - SPA routing support

### Frontend Configuration: `/rag-from-scratch/frontend/vercel.json`
- **Purpose**: Frontend-specific optimizations
- **Key Features**:
  - Static asset caching
  - Content Security Policy
  - Clean URLs and trailing slash handling

### Environment Template: `/.env.vercel.example`
- **Purpose**: Template for required environment variables
- **Usage**: Copy and configure in Vercel dashboard

## Deployment Steps

### 1. Repository Setup

Ensure your repository structure matches:
```
RAG for LLMs- A Survey/
├── vercel.json                 # Main config
├── .vercelignore              # Deployment exclusions
├── .env.vercel.example        # Environment template
└── rag-from-scratch/
    └── frontend/
        ├── vercel.json        # Frontend config
        ├── package.json       # Updated with Vercel scripts
        ├── vite.config.ts     # Vite configuration
        └── src/               # Source code
```

### 2. Vercel Dashboard Configuration

1. **Import Project**
   - Go to [Vercel Dashboard](https://vercel.com/dashboard)
   - Click "New Project"
   - Import from GitHub (recommended for auto-deployment)

2. **Build Configuration**
   ```
   Framework Preset: Vite
   Root Directory: rag-from-scratch/frontend
   Build Command: npm run build
   Output Directory: dist
   Install Command: npm install
   ```

3. **Environment Variables**
   Set in Project Settings > Environment Variables:
   ```
   VITE_API_URL=https://your-backend-api.com
   NODE_ENV=production
   ```

### 3. Domain Configuration

1. **Custom Domain** (Optional)
   - Go to Project Settings > Domains
   - Add your custom domain
   - Configure DNS records as instructed

2. **SSL Certificate**
   - Automatically provided by Vercel
   - Supports custom domains with automatic HTTPS

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API endpoint | `https://api.yourapp.com` |
| `NODE_ENV` | Node environment | `production` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_TIMEOUT` | API request timeout | `30000` |
| `VITE_ENABLE_AUTH` | Enable authentication | `true` |
| `VITE_ENABLE_STREAMING` | Enable streaming responses | `true` |

### Setting Environment Variables

**Via Vercel CLI:**
```bash
vercel env add VITE_API_URL production
```

**Via Dashboard:**
1. Go to Project Settings
2. Navigate to Environment Variables
3. Add key-value pairs
4. Select environments (Production, Preview, Development)

## Build Optimization

### Performance Features

1. **Static Asset Caching**
   - Assets cached for 1 year with immutable headers
   - Configured in frontend `vercel.json`

2. **Bundle Analysis**
   - Run `npm run build:analyze` to analyze bundle size
   - Monitor bundle size warnings

3. **Code Splitting**
   - Automatic code splitting via Vite
   - Route-based splitting implemented

### Build Scripts

```json
{
  "build": "vite build",
  "build:vercel": "vite build --mode production",
  "build:analyze": "vite build --mode production && npx vite-bundle-analyzer",
  "preview:vercel": "vite preview --host 0.0.0.0 --port 3000"
}
```

## Security Configuration

### Headers Applied

1. **Security Headers**
   - `X-Content-Type-Options: nosniff`
   - `X-Frame-Options: DENY`
   - `X-XSS-Protection: 1; mode=block`
   - `Referrer-Policy: strict-origin-when-cross-origin`

2. **Content Security Policy**
   - Restricts resource loading
   - Allows API connections to configured backend
   - Prevents XSS attacks

3. **CORS Configuration**
   - API proxy with proper CORS headers
   - Supports preflight requests
   - Configurable origins

## API Integration

### Proxy Configuration

The frontend proxies API requests to avoid CORS issues:

```json
{
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "$VITE_API_URL/api/$1"
    }
  ]
}
```

### Backend Requirements

Your backend API must:

1. **Accept Requests** from Vercel domain
2. **Configure CORS** for your Vercel URL
3. **Handle Preflight** requests (OPTIONS method)
4. **Use HTTPS** in production

Example backend CORS configuration:
```python
# FastAPI example
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-app.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Deployment Workflows

### Automatic Deployment

1. **GitHub Integration**
   - Pushes to `main` branch trigger production deployments
   - Pull requests create preview deployments
   - Automatic deployment status updates

2. **Preview Deployments**
   - Every push to feature branches creates a preview
   - Unique URL for each preview
   - Perfect for testing and reviews

### Manual Deployment

```bash
# Deploy to preview
vercel

# Deploy to production
vercel --prod

# Deploy specific directory
vercel rag-from-scratch/frontend --prod
```

## Monitoring & Analytics

### Built-in Analytics

Vercel provides:
- **Core Web Vitals** monitoring
- **Function execution** metrics
- **Bandwidth usage** tracking
- **Error logging** and alerts

### Access Analytics

1. Go to Project Dashboard
2. Navigate to Analytics tab
3. Monitor performance metrics
4. Set up alerts for issues

## Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   # Check build locally
   cd rag-from-scratch/frontend
   npm run build:vercel
   
   # Verify dependencies
   npm install
   npm audit fix
   ```

2. **API Connection Issues**
   - Verify `VITE_API_URL` is set correctly
   - Check backend CORS configuration
   - Ensure backend accepts HTTPS requests

3. **Environment Variable Problems**
   - Variables must be prefixed with `VITE_`
   - Restart deployment after changing variables
   - Check variable spelling and casing

4. **Routing Issues**
   - SPA routing configured in `vercel.json`
   - All routes fallback to `index.html`
   - Verify React Router configuration

### Debug Commands

```bash
# Check deployment logs
vercel logs

# Inspect build output
vercel build

# Test locally with production build
npm run build && npm run preview:vercel
```

### Support Resources

- [Vercel Documentation](https://vercel.com/docs)
- [Vite Deployment Guide](https://vitejs.dev/guide/static-deploy.html#vercel)
- [React Router with Vercel](https://vercel.com/guides/deploying-react-with-vercel#routing)

## Best Practices

### Performance
1. **Optimize Images** - Use appropriate formats and sizes
2. **Minimize Bundle Size** - Regular bundle analysis
3. **Lazy Loading** - Implement for routes and components
4. **Caching Strategy** - Leverage Vercel's edge caching

### Security
1. **Environment Variables** - Never commit sensitive data
2. **Content Security Policy** - Regularly review and update
3. **Dependency Updates** - Keep dependencies current
4. **HTTPS Only** - Always use HTTPS in production

### Maintenance
1. **Regular Deployments** - Keep deployments frequent and small
2. **Monitor Metrics** - Watch Core Web Vitals and errors
3. **Preview Testing** - Test features in preview deployments
4. **Documentation** - Keep deployment docs updated

## Advanced Configuration

### Custom Build Pipeline

For complex build requirements, create a custom build script:

```javascript
// build.js
const { build } = require('vite');
const { execSync } = require('child_process');

async function buildWithOptimizations() {
  // Pre-build optimizations
  execSync('npm run lint:fix');
  
  // Build with custom config
  await build({
    mode: 'production',
    define: {
      __BUILD_TIME__: JSON.stringify(new Date().toISOString())
    }
  });
  
  // Post-build optimizations
  console.log('Build completed with optimizations');
}

buildWithOptimizations();
```

### Multi-Environment Setup

Configure different environments:

```json
{
  "env": {
    "VITE_API_URL": {
      "production": "https://api.production.com",
      "preview": "https://api.staging.com",
      "development": "http://localhost:8000"
    }
  }
}
```

---

## Summary

This deployment configuration provides:

✅ **Production-Ready Setup** - Optimized for performance and security  
✅ **Auto-Deployment** - GitHub integration for seamless deployments  
✅ **API Integration** - Proper proxy configuration for backend communication  
✅ **Security Headers** - Comprehensive security configuration  
✅ **Performance Optimization** - Caching, compression, and bundle optimization  
✅ **Environment Management** - Flexible environment variable configuration  

Your RAG frontend is now ready for production deployment on Vercel with enterprise-grade configuration and optimization.