/**
 * This file is part of LevenbergMarquardt Solver.
 *
 * Copyright (C) 2018-2020 Dongsheng Yang <ydsf16@buaa.edu.cn> (Beihang University)
 * For more information see <https://github.com/ydsf16/LevenbergMarquardt>
 *
 * LevenbergMarquardt is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * LevenbergMarquardt is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with LevenbergMarquardt. If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Cholesky>
#include <chrono>

/* 计时类 */
class Runtimer{
public:
    inline void start()
    {
        t_s_  = std::chrono::steady_clock::now();
    }
    
    inline void stop()
    {
        t_e_ = std::chrono::steady_clock::now();
    }
    
    inline double duration()
    {
        return std::chrono::duration_cast<std::chrono::duration<double>>(t_e_ - t_s_).count() * 1000.0;
    }
    
private:
    std::chrono::steady_clock::time_point t_s_; //start time ponit
    std::chrono::steady_clock::time_point t_e_; //stop time point
};

/*  优化方程 */
class LevenbergMarquardt{
public:
    LevenbergMarquardt(double* a, double* b, double* c): // 构造函数
    a_(a), b_(b), c_(c) // 初始化参数
    {
        // 默认参数
        epsilon_1_ = 1e-6;
        epsilon_2_ = 1e-6;
        max_iter_ = 50;
        is_out_ = true;
    }
    
    // 设置参数,参数由外部传入
    void setParameters(double epsilon_1, double epsilon_2, int max_iter, bool is_out)
    {
        epsilon_1_ = epsilon_1;
        epsilon_2_ = epsilon_2;
        max_iter_ = max_iter;
        is_out_ = is_out;
    }
    
    void addObservation(const double& x, const double& y)
    {
        obs_.push_back(Eigen::Vector2d(x, y));
    }
    
    // 计算雅克比矩阵和残差
    void calcJ_fx()
    {
        J_ .resize(obs_.size(), 3); // 雅可比矩阵的大小为：观测数N * 参数数3
        fx_.resize(obs_.size(), 1); // 残差的大小为：观测数N * 1
        
        for ( size_t i = 0; i < obs_.size(); i ++)
        {
            const Eigen::Vector2d& ob = obs_.at(i);
            const double& x = ob(0);
            const double& y = ob(1);
            double j1 = -x*x*exp(*a_ * x*x + *b_*x + *c_); // 残差fx对a的偏导
            double j2 = -x*exp(*a_ * x*x + *b_*x + *c_); // 残差fx对b的偏导
            double j3 = -exp(*a_ * x*x + *b_*x + *c_); // 残差fx对c的偏导
            J_(i, 0 ) = j1;
            J_(i, 1) = j2;
            J_(i, 2) = j3;
            fx_(i, 0) = y - exp( *a_ *x*x + *b_*x +*c_); 
        }
    }
    
    void calcH_g()
    {
        H_ = J_.transpose() * J_; // H矩阵大小为：参数数3 * 参数数3
        g_ = -J_.transpose() * fx_; // g向量大小为：参数数3 * 1，注意负号
    }
        
    double getCost()
    {
        Eigen::MatrixXd cost= fx_.transpose() * fx_;
        return cost(0,0);
    }
    
    // 计算目标函数,即残差的平方和
    double F(double a, double b, double c)
    {
        Eigen::MatrixXd fx;
        fx.resize(obs_.size(), 1); // 残差的大小为：观测数N * 1
        
        for ( size_t i = 0; i < obs_.size(); i ++)
        {
            const Eigen::Vector2d& ob = obs_.at(i);
            const double& x = ob(0);
            const double& y = ob(1);
            fx(i, 0) = y - exp( a *x*x + b*x +c);
        }
        Eigen::MatrixXd F = 0.5 * fx.transpose() * fx; // 目标函数, 1*1矩阵
        return F(0,0);
    }
    
    double L0_L( Eigen::Vector3d& h)
    {
            // Eigen::MatrixXd L = -h.transpose() * J_.transpose() * fx_ - 0.5 * h.transpose() * J_.transpose() * J_ * h;
            Eigen::MatrixXd L = h.transpose() * g_ - 0.5 * h.transpose() * H_ * h; // 数学推导上与上面等价
            // Eigen::MatrixXd L = h.transpose() * g_ - mu * h.transpose() * h; // 原公式，这里的mu是H矩阵对角元素的最大值，收敛变慢
            // std::cout << "L0_L: " << L(0,0) << " " << L1(0,0) << std::endl;
            return L(0,0);
    }

    void solve()
    {
        int k = 0;
        double nu = 2.0;
        calcJ_fx(); // 计算雅克比矩阵和残差
        calcH_g(); // 计算H矩阵和g向量

        /* 判断是否收敛
        g_.lpNorm<Eigen::Infinity>() 表示g向量的无穷范数，即g向量中绝对值最大的元素
        epsilon_1_ 表示残差的阈值
        */ 
        bool found = ( g_.lpNorm<Eigen::Infinity>() < epsilon_1_ );
        
        std::vector<double> A; // 用于存储H矩阵的对角元素
        A.push_back( H_(0, 0) );
        A.push_back( H_(1, 1) );
        A.push_back( H_(2,2) );
        auto max_p = std::max_element(A.begin(), A.end()); // 找到H矩阵对角元素的最大值
        mu = *max_p; // mu为H矩阵对角元素的最大值
        
        double sumt =0;

        while ( !found && k < max_iter_) // 迭代求解，直到收敛或者达到最大迭代次数
        {
            Runtimer t;
            t.start();
            
            k = k + 1; // 迭代次数加1
            Eigen::Matrix3d G = H_ + mu * Eigen::Matrix3d::Identity(); // G矩阵，Identity()表示单位矩阵
            Eigen::Vector3d h = G.ldlt().solve(g_); // 求解方程Gh = g(即求解h), ldlt()表示LDLT分解
            
            if( h.norm() <= epsilon_2_ * ( sqrt(*a_**a_ + *b_**b_ + *c_**c_ ) + epsilon_2_ ) )
                found = true; // 如果步长太小，直接认为收敛
            else
            {
                // 更新参数
                double na = *a_ + h(0);
                double nb = *b_ + h(1);
                double nc = *c_ + h(2);
                
                double rho =( F(*a_, *b_, *c_) - F(na, nb, nc) )  / L0_L(h);

                if( rho > 0) // 如果rho > 0,则更新参数
                {
                    *a_ = na;
                    *b_ = nb;
                    *c_ = nc;
                    calcJ_fx();
                    calcH_g();
                                      
                    found = ( g_.lpNorm<Eigen::Infinity>() < epsilon_1_ );
                    mu = mu * std::max<double>(0.33, 1 - std::pow(2*rho -1, 3));
                    nu = 2.0;
                }
                else // 如果rho <= 0,则不更新参数
                {
                    mu = mu * nu; 
                    nu = 2*nu;
                }// if rho > 0
            }// if step is too small
            
            t.stop();
            if( is_out_ )
            {
                std::cout << "Iter: " << std::left <<std::setw(3) << k << " Result: "<< std::left <<std::setw(10)  << *a_ << " " << std::left <<std::setw(10)  << *b_ << " " << std::left <<std::setw(10) << *c_ << 
                " step: " << std::left <<std::setw(14) << h.norm() << " cost: "<< std::left <<std::setw(14)  << getCost() << " time: " << std::left <<std::setw(14) << t.duration()  <<
                " total_time: "<< std::left <<std::setw(14) << (sumt += t.duration()) << std::endl;
            }   
        } // while
        
        if( found  == true)
            std::cout << "\nConverged\n\n";
        else
            std::cout << "\nDiverged\n\n";
        
    }//function 
    
    double mu;
    
    Eigen::MatrixXd fx_; 
    Eigen::MatrixXd J_; // 雅克比矩阵
    Eigen::Matrix3d H_; // H矩阵
    Eigen::Vector3d g_; // g向量 
    
    std::vector< Eigen::Vector2d> obs_; // 观测
   
   /* 要求的三个参数 */
   double* a_, *b_, *c_;
    
    /* parameters */
    double epsilon_1_, epsilon_2_;
    int max_iter_;
    bool is_out_;
};//class LevenbergMarquardt
int main(int argc, char **argv) {
    const double aa = 0.1, bb = 0.5, cc = 2; // 实际方程的参数
    double a =0.0, b=0.0, c=0.0; // 初值
    
    /* 构造问题 */
    LevenbergMarquardt lm(&a, &b, &c);
    lm.setParameters(1e-10, 1e-10, 100, true);
    
    /* 制造数据 */
    const size_t N = 100; //数据个数
    cv::RNG rng(cv::getTickCount()); // 随机数生成器
    for( size_t i = 0; i < N; i ++)
    {
        /* 生产带有高斯噪声的数据　*/
        double x = rng.uniform(0.0, 1.0) ; // 服从均匀分布
        double y = exp(aa*x*x + bb*x + cc) + rng.gaussian(0.05);
        
        /* 添加到观测中　*/
        lm.addObservation(x, y);
    }
    /* 用LM法求解 */
    lm.solve();
    
    return 0;
}
