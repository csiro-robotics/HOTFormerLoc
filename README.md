
# HOTFormerLoc
Hierarchical Octree Transformer for Versatile Lidar Place Recognition Across Ground and Aerial Views

# Run Website Locally
## Environment Variables

## Environment Setup
### Install node.js
The following works for Ubuntu 22.04, but check here for alternatives if you run into errors: https://nodejs.org/en/download/package-manager
```
# installs nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash

# download and install Node.js (you may need to restart the terminal)
nvm install 22

# verifies the right Node.js version is in the environment
node -v # should print `v22.12.0`

# verifies the right npm version is in the environment
npm -v # should print `10.9.0`
```
### Install packages
`npm install`
## Running Locally
`npm run dev`
## Deploying to GitHub (may take a while)
`npm run deploy`