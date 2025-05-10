wget -P external https://github.com/microsoft/onnxruntime/releases/download/v1.21.1/onnxruntime-linux-x64-gpu-1.21.1.tgz
cd external
tar -xvzf onnxruntime-linux-x64-gpu-1.21.1.tgz

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cudnn-cuda-12


class Solution {
public:
    int minTimeToReach(vector<vector<int>>& moveTime) {
        int n = moveTime.size();
        int m = moveTime[0].size();

        struct point {
            int x, y;
            int priority;

            point(int x_, int y_, int p) : x(x_), y(y_), priority(p) {}
        };

        struct compare {
            bool operator() (const point &p1, const point &p2) {
                return p1.priority > p2.priority;
            }
        }

        vector<vector<bool>> isVis(n, vector<bool> (m, 0));
        priority_queue<point, vector<point>, compare> fron; fron.push_back(point(0, 0, 0));
        isVis[0][0] = 1;

        while(fron.size()) {
            auto pt = fron.top(); 
            int u = pt.x;
            int v = pt.y;
            int p = pt.priority;
            fron.pop();

            if((u == n-1) && (v == m-1)) return p;

            // try going up
            if((u>0) && (isVis[u-1][v] == 0)) {
                isVis[u-1][v] = 1;
                int t = 1 + min(p, moveTime[u-1][v]);
                fron.insert(point(u-1, v, t));
            }

            // try going left
            if((v>0) && (isVis[u][v-1] == 0)) {
                isVis[uq][v-1] = 1;
                int t = 1 + min(p, moveTime[u][v-1]);
                fron.insert(point(u, v-1, t));
            }

            // try going right
            if((v<(m-1)) && (isVis[u][v+1] == 0)) {
                isVis[u][v+1] = 1;
                int t = 1 + min(p, moveTime[u][v+1]);
                fron.insert(point(u, v+1, t));
            }

            // try going down
            if((u<(n-1)) && (isVis[u+1][v] == 0)) {
                isVis[u+1][v] = 1;
                int t = 1 + min(p, moveTime[u+1][v]);
                fron.insert(point(u+1, v, t));
            }
        }

        return -1;
    }
};