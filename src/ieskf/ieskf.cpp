#include "ieskf/ieskf.h"

namespace kf
{
    /// @brief 该函数计算输入向量的右雅可比矩阵（right Jacobian）
    /// @param inp Eigen::Vector3d类型的三维向量
    /// @return 
    Eigen::Matrix3d rightJacobian(const Eigen::Vector3d &inp)
    {
        return Sophus::SO3d::leftJacobian(inp).transpose();//返回左雅可比矩阵的转置
    }

    /// @brief 该函数重载了+=运算符，用于更新状态向量
    /// @param delta Eigen::Matrix<double, 23, 1>类型的向量，表示状态增量
    void State::operator+=(const Vector23d &delta)
    {
        pos += delta.segment<3>(0);//更新位置
        rot *= Sophus::SO3d::exp(delta.segment<3>(3)).matrix();//更新旋转
        rot_ext *= Sophus::SO3d::exp(delta.segment<3>(6)).matrix();//更新外部旋转
        pos_ext += delta.segment<3>(9);//更新外部位置
        vel += delta.segment<3>(12);//更新速度
        bg += delta.segment<3>(15);//更新陀螺仪偏差
        ba += delta.segment<3>(18);//更新加速度计偏差
        g = Sophus::SO3d::exp(getBx() * delta.segment<2>(21)).matrix() * g;//更新重力,这里是两个元素完成差值
    }

    /// @brief 该函数重载了-运算符，用于计算两个状态向量之间的差
    /// @param other State类型的状态向量
    void State::operator+=(const Vector24d &delta)
    {
        pos += delta.segment<3>(0);
        rot *= Sophus::SO3d::exp(delta.segment<3>(3)).matrix();
        rot_ext *= Sophus::SO3d::exp(delta.segment<3>(6)).matrix();
        pos_ext += delta.segment<3>(9);
        vel += delta.segment<3>(12);
        bg += delta.segment<3>(15);
        ba += delta.segment<3>(18);
        g = Sophus::SO3d::exp(delta.segment<3>(21)).matrix() * g;// 更新重力，这里直接是三个元素的差值
    }

    /// @brief 该函数重载了-运算符，用于计算当前状态与另一个状态之间的差值
    /// @param other State类型，总共也是24个元素
    /// @return 
    Vector23d State::operator-(const State &other)
    {
        Vector23d delta = Vector23d::Zero();
        delta.segment<3>(0) = pos - other.pos;
        delta.segment<3>(3) = Sophus::SO3d(other.rot.transpose() * rot).log();
        delta.segment<3>(6) = Sophus::SO3d(other.rot_ext.transpose() * rot_ext).log();
        delta.segment<3>(9) = pos_ext - other.pos_ext;
        delta.segment<3>(12) = vel - other.vel;
        delta.segment<3>(15) = bg - other.bg;
        delta.segment<3>(18) = ba - other.ba;
        
        //  这个矩阵的作用是用于叉乘操作，等价于 \hat{g} ⋅ other.g等同于g×other.g。然后通过 norm() 函数计算叉乘结果的范数，表示两个向量形成的平行四边形的面积
        //  vsin​=||g×other.g||=||g|| ||other.g||sin(θ)
        double v_sin = (Sophus::SO3d::hat(g) * other.g).norm();//计算两个向量的叉乘，标准重力加速度和当前重力加速度的叉乘，得到SIN值
        // vcos​=g⋅other.g=||g|| ||other.g||cos(θ)
        double v_cos = g.transpose() * other.g;//计算两个向量的点乘,得到两个向量的COS值


        double theta = std::atan2(v_sin, v_cos);//计算两个向量的夹角
        Eigen::Vector2d res;
        if (v_sin < 1e-11)//如果sin值小于1e-11,则认为两个向量平行
        {
            if (std::fabs(theta) > 1e-11)//如果夹角大于1e-11,认为两个向量方向相反
            {
                res << 3.1415926, 0;
            }
            else
            {
                res << 0, 0;
            }
        }
        else
        {
            //具体公式为角度差*重力状态向量转置（2*3）*传入的重力向量的反对称矩阵（3*3）*当前重力向量（3*1）/叉乘结果的范数
            res = theta / v_sin * other.getBx().transpose() * Sophus::SO3d::hat(other.g) * g;
        }
        delta.segment<2>(21) = res;//将计算得到的重力向量差值赋值给delta的后两个元素
        return delta;
    }

    /// @brief 该函数计算状态向量的Bx矩阵，用来描述重力方向在某些参考坐标系中的表示,这里是在标准重力下的表示
    /// @return 
    Matrix3x2d State::getBx() const
    {
        Matrix3x2d res;
        res << -g[1], -g[2],
            GRAVITY - g[1] * g[1] / (GRAVITY + g[0]), -g[2] * g[1] / (GRAVITY + g[0]),
            -g[2] * g[1] / (GRAVITY + g[0]), GRAVITY - g[2] * g[2] / (GRAVITY + g[0]);//元素通过重力向量的分量和常数GRAVITY计算得到
        res /= GRAVITY;//归一化
        return res;
    }

    /// @brief 该函数基于当前的重力向量和Bx矩阵计算Mx矩阵,这个矩阵描述系统在受到外部扰动（例如，传感器噪声或外力）时的响应。
    /// @return 
    Matrix3x2d State::getMx() const
    {

        return -Sophus::SO3d::hat(g) * getBx();
    }

    /// @brief brief 该函数基于当前的重力向量和Bx矩阵计算Mx矩阵,这个矩阵描述系统在受到外部扰动（例如，传感器噪声或外力）时的响应。
    /// @param res 这个向量是一个2维向量，表示重力向量的差值
    /// @return 
    Matrix3x2d State::getMx(const Eigen::Vector2d &res) const
    {
        Matrix3x2d bx = getBx();
        Eigen::Vector3d bu = bx * res;//转到标准重力下
        //下面表示在旋转群SO(3)中，通过李代数的扰动  u，对向量 g 的反对称矩阵进行变换，并结合左雅可比矩阵的转置作用，最终作用在向量 b x ​ 上
        return -Sophus::SO3d::exp(bu).matrix() * Sophus::SO3d::hat(g) * Sophus::SO3d::leftJacobian(bu).transpose() * bx;
    }

    /// @brief 用于描述系统的某些状态对重力变化的敏感度，用于描述getMx这种扰动的反向传播
    /// @return 
    Matrix2x3d State::getNx() const
    {
        return 1 / GRAVITY / GRAVITY * getBx().transpose() * Sophus::SO3d::hat(g);
    }

    IESKF::IESKF() = default;

    /// @brief 该函数用于初始化IESKF滤波器
    /// @param inp 输入的状态向量
    /// @param dt 时间间隔
    /// @param Q 状态转移矩阵
    void IESKF::predict(const Input &inp, double dt, const Matrix12d &Q)
    {
        Vector24d delta = Vector24d::Zero();
        //计算状态增量
        delta.segment<3>(0) = x_.vel * dt;//位置增量
        delta.segment<3>(3) = (inp.gyro - x_.bg) * dt;//旋转增量
        delta.segment<3>(12) = (x_.rot * (inp.acc - x_.ba) + x_.g) * dt;//速度增量
        //更新状态矩阵F_和G_
        F_.setIdentity();//使用右雅可比矩阵和输入的加速度和陀螺仪数据计算状态转移矩阵F_
        F_.block<3, 3>(0, 12) = Eigen::Matrix3d::Identity() * dt;//位置与速度的耦合
        F_.block<3, 3>(3, 3) = Sophus::SO3d::exp(-(inp.gyro - x_.bg) * dt).matrix();//旋转状态自身的变化
        F_.block<3, 3>(3, 15) = -rightJacobian((inp.gyro - x_.bg) * dt) * dt;//旋转状态与陀螺仪偏置的耦合
        F_.block<3, 3>(12, 3) = -x_.rot * Sophus::SO3d::hat(inp.acc - x_.ba) * dt;//加速度对旋转状态的影响
        F_.block<3, 3>(12, 18) = -x_.rot * dt;//加速度对加速度计偏置的影响
        F_.block<3, 2>(12, 21) = x_.getMx() * dt;//加速度对重力变化的影响
        F_.block<2, 2>(21, 21) = x_.getNx() * x_.getMx();//重力变化自身的变化

        G_.setZero();//陀螺仪，加速度，陀螺仪偏置，加速度计偏置的噪声矩阵
        G_.block<3, 3>(3, 0) = -rightJacobian((inp.gyro - x_.bg) * dt) * dt;//旋转状态与陀螺仪噪声的耦合
        G_.block<3, 3>(12, 3) = -x_.rot * dt;//加速度对加速度计噪声的影响
        G_.block<3, 3>(15, 6) = Eigen::Matrix3d::Identity() * dt;//陀螺仪偏置噪声
        G_.block<3, 3>(18, 9) = Eigen::Matrix3d::Identity() * dt;//加速度计偏置噪声
        x_ += delta;
        P_ = F_ * P_ * F_.transpose() + G_ * Q * G_.transpose();//更新协方差矩阵
    }

    /// @brief 该函数用于更新IESKF滤波器
    void IESKF::update()
    {
        State predict_x = x_;//保存当前状态向量
        SharedState shared_data;
        shared_data.iter_num = 0;
        Vector23d delta = Vector23d::Zero();
        for (size_t i = 0; i < max_iter_; i++)
        {
            func_(x_, shared_data);//更新状态向量
            H_.setZero();
            b_.setZero();
            delta = x_ - predict_x;//计算状态向量的差值
            Matrix23d J = Matrix23d::Identity();
            J.block<3, 3>(3, 3) = rightJacobian(delta.segment<3>(3));
            J.block<3, 3>(6, 6) = rightJacobian(delta.segment<3>(6));
            J.block<2, 2>(21, 21) = x_.getNx() * predict_x.getMx(delta.segment<2>(21));//计算雅可比矩阵
            b_ += (J.transpose() * P_.inverse() * delta);//更新b矩阵
            H_ += (J.transpose() * P_.inverse() * J);//更新H矩阵
            H_.block<12, 12>(0, 0) += shared_data.H;//更新H矩阵
            b_.block<12, 1>(0, 0) += shared_data.b;//更新b矩阵
            delta = -H_.inverse() * b_;
            x_ += delta;
            shared_data.iter_num += 1;
            if (delta.maxCoeff() < eps_)
                break;
        }
        Matrix23d L = Matrix23d::Identity();
        L.block<3, 3>(3, 3) = rightJacobian(delta.segment<3>(3));
        L.block<3, 3>(6, 6) = rightJacobian(delta.segment<3>(6));
        L.block<2, 2>(21, 21) = x_.getNx() * predict_x.getMx(delta.segment<2>(21));
        P_ = L * H_.inverse() * L.transpose();
    }

} // namespace kf
