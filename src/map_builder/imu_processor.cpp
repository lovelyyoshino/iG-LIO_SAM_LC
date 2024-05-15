#include "map_builder/imu_processor.h"

namespace lio
{
    IMUProcessor::IMUProcessor(std::shared_ptr<kf::IESKF> kf)
        : kf_(kf), init_count_(0), last_lidar_time_end_(0.0),
          mean_acc_(Eigen::Vector3d::Zero()), mean_gyro_(Eigen::Vector3d::Zero()),
          last_acc_(Eigen::Vector3d::Zero()), last_gyro_(Eigen::Vector3d::Zero()),
          rot_ext_(Eigen::Matrix3d::Identity()), pos_ext_(Eigen::Vector3d::Zero())
    {
    }

    void IMUProcessor::setExtParams(Eigen::Matrix3d &rot_ext, Eigen::Vector3d &pos_ext)
    {
        rot_ext_ = rot_ext;
        pos_ext_ = pos_ext;
    }

    void IMUProcessor::setCov(Eigen::Vector3d gyro_cov, Eigen::Vector3d acc_cov, Eigen::Vector3d gyro_bias_cov, Eigen::Vector3d acc_bias_cov)
    {
        gyro_cov_ = gyro_cov;
        acc_cov_ = acc_cov;
        gyro_bias_cov_ = gyro_bias_cov;
        acc_bias_cov_ = acc_bias_cov;
    }

    void IMUProcessor::setCov(double gyro_cov, double acc_cov, double gyro_bias_cov, double acc_bias_cov)
    {
        gyro_cov_ = Eigen::Vector3d(gyro_cov, gyro_cov, gyro_cov);
        acc_cov_ = Eigen::Vector3d(acc_cov, acc_cov, acc_cov);
        gyro_bias_cov_ = Eigen::Vector3d(gyro_bias_cov, gyro_bias_cov, gyro_bias_cov);
        acc_bias_cov_ = Eigen::Vector3d(acc_bias_cov, acc_bias_cov, acc_bias_cov);
    }

    /// @brief IMU 初始化函数，用于静态估计重力和角速度偏置。
    /// @param meas 包含IMU和点云数据的 MeasureGroup 对象。
    void IMUProcessor::init(const MeasureGroup &meas)
    {
        //检查IMU数据是否为空
        if (meas.imus.empty())
            return;

        // 静态初始化，估计重力和角速度偏置

        for (const auto &imu : meas.imus)
        {
            //遍历 meas.imus 中的每个 IMU 数据，更新加速度和陀螺仪的平均值.使用逐步平均算法计算平均加速度和平均陀螺仪值
            init_count_++;
            mean_acc_ += (imu.acc - mean_acc_) / init_count_;
            mean_gyro_ += (imu.gyro - mean_gyro_) / init_count_;
        }
        if (init_count_ < max_init_count_)
            return;
        init_flag_ = true;

        // 设置初始化状态
        kf::State state = kf_->x();
        state.rot_ext = rot_ext_;
        state.pos_ext = pos_ext_;
        state.bg = mean_gyro_;
        // TODO: 对于初始为非水平放置的情况进行重力对齐
        if (align_gravity_)
        {
            //如果需要重力对齐，根据平均加速度计算初始旋转。

            // Eigen::Vector3d euler_angles = Eigen::Quaterniond::FromTwoVectors((-mean_acc_).normalized(), Eigen::Vector3d(0.0, 0.0, -1.0)).matrix().eulerAngles(0, 1, 2);
            // Eigen::AngleAxisd roll(euler_angles(0), Eigen::Vector3d::UnitX());
            // Eigen::AngleAxisd pitch(euler_angles(1), Eigen::Vector3d::UnitY());
            // Eigen::AngleAxisd yaw(euler_angles(2), Eigen::Vector3d::UnitZ());
            // std::cout << euler_angles(2) << std::endl;

            // Eigen::Matrix3d rot = (yaw * pitch * roll).matrix();
            //将初始旋转设置为从 -mean_acc_ 向量到 (0,0,−1) 方向的旋转
            state.rot = (Eigen::Quaterniond::FromTwoVectors((-mean_acc_).normalized(), Eigen::Vector3d(0.0, 0.0, -1.0)).matrix());
            state.initG(Eigen::Vector3d(0, 0, -1.0));
        }
        else
        {
            state.initG(-mean_acc_);
        }

        //  v2 = q * v1

        kf_->change_x(state);//更新状态向量

        // 初始化噪声的协方差矩阵,设置状态协方差矩阵的初始值，表示初始状态的不确定性
        kf::Matrix23d init_P = kf_->P();
        init_P.setIdentity();
        init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
        init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
        init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
        init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;
        init_P(21, 21) = init_P(22, 22) = 0.00001;
        kf_->change_P(init_P);

        last_imu_ = meas.imus.back();
    }

    /// @brief 矫正点云畸变，根据IMU数据对点云进行去畸变处理
    /// @param meas 包含IMU和点云数据的 MeasureGroup 对象
    /// @param out 输出的矫正后的点云，类型为 PointCloudXYZI::Ptr
    void IMUProcessor::undistortPointcloud(const MeasureGroup &meas, PointCloudXYZI::Ptr &out)
    {
        //将 IMU 数据转换为双端队列，并添加最后一个 IMU 数据到队首
        std::deque<IMU> v_imus(meas.imus.begin(), meas.imus.end());
        v_imus.push_front(last_imu_);
        const double imu_time_begin = v_imus.front().timestamp;
        const double imu_time_end = v_imus.back().timestamp;
        const double lidar_time_begin = meas.lidar_time_begin;
        const double lidar_time_end = meas.lidar_time_end;

        out = meas.lidar;
        //按曲率排序点云，以确保点云按照扫描顺序处理
        std::sort(out->points.begin(), out->points.end(), [](PointType &p1, PointType &p2) -> bool
                  { return p1.curvature < p2.curvature; });
        //初始化状态对象，并将其添加到 imu_poses_ 容器中
        kf::State state = kf_->x();
        imu_poses_.clear();
        imu_poses_.emplace_back(0.0, last_acc_, last_gyro_, state.vel, state.pos, state.rot);

        Eigen::Vector3d acc_val, gyro_val;
        double dt = 0.0;
        Q_.setIdentity();

        kf::Input inp;
        // 计算每一帧IMU的位姿
        for (auto it_imu = v_imus.begin(); it_imu < (v_imus.end() - 1); it_imu++)
        {
            IMU &head = *it_imu;
            IMU &tail = *(it_imu + 1);
            if (tail.timestamp < last_lidar_time_end_)
                continue;
            //对 IMU 数据进行处理，计算加速度和角速度的中值
            gyro_val = 0.5 * (head.gyro + tail.gyro);
            acc_val = 0.5 * (head.acc + head.acc);
            // normalize acc
            acc_val = acc_val * 9.81 / mean_acc_.norm();

            if (head.timestamp < last_lidar_time_end_)
                dt = tail.timestamp - last_lidar_time_end_;
            else
                dt = tail.timestamp - head.timestamp;
            //归一化加速度，并根据时间间隔 dt 预测下一时刻的状态
            Q_.block<3, 3>(0, 0).diagonal() = gyro_cov_;
            Q_.block<3, 3>(3, 3).diagonal() = acc_cov_;
            Q_.block<3, 3>(6, 6).diagonal() = gyro_bias_cov_;
            Q_.block<3, 3>(9, 9).diagonal() = acc_bias_cov_;
            inp.acc = acc_val;
            inp.gyro = gyro_val;
            kf_->predict(inp, dt, Q_);

            state = kf_->x();

            last_gyro_ = gyro_val - state.bg;
            last_acc_ = state.rot * (acc_val - state.ba);
            last_acc_ += state.g;

            double offset = tail.timestamp - lidar_time_begin;
            imu_poses_.emplace_back(offset, last_acc_, last_gyro_, state.vel, state.pos, state.rot);
        }

        // 计算最后一个点云的位姿
        // double sign = lidar_time_end > imu_time_end ? 1.0 : -1.0;
        dt = lidar_time_end - imu_time_end;
        kf_->predict(inp, dt, Q_);

        last_imu_ = v_imus.back();
        last_lidar_time_end_ = lidar_time_end;

        state = kf_->x();
        Eigen::Matrix3d cur_rot = state.rot;
        Eigen::Vector3d cur_pos = state.pos;
        Eigen::Matrix3d cur_rot_ext = state.rot_ext;
        Eigen::Vector3d cur_pos_ext = state.pos_ext;

        // 畸变矫正
        auto it_pcl = out->points.end() - 1;
        //从最后一个点开始，逐点处理点云数据，根据 IMU 数据进行去畸变
        for (auto it_kp = imu_poses_.end() - 1; it_kp != imu_poses_.begin(); it_kp--)
        {
            auto head = it_kp - 1;
            auto tail = it_kp;

            Eigen::Matrix3d imu_rot = head->rot;
            Eigen::Vector3d imu_pos = head->pos;
            Eigen::Vector3d imu_vel = head->vel;
            Eigen::Vector3d imu_acc = tail->acc;
            Eigen::Vector3d imu_gyro = tail->gyro;

            //计算点云点在时间上的位姿变化，并将其转换到当前时刻的位姿
            for (; it_pcl->curvature / double(1000) > head->offset; it_pcl--)
            {
                dt = it_pcl->curvature / double(1000) - head->offset;
                Eigen::Vector3d point(it_pcl->x, it_pcl->y, it_pcl->z);
                Eigen::Matrix3d point_rot = imu_rot * Sophus::SO3d::exp(imu_gyro * dt).matrix();
                Eigen::Vector3d point_pos = imu_pos + imu_vel * dt + 0.5 * imu_acc * dt * dt;
                // T_l_b * T_j_w * T_w_i * T_b_l * p
                Eigen::Vector3d p_compensate = cur_rot_ext.transpose() * (cur_rot.transpose() * (point_rot * (cur_rot_ext * point + cur_pos_ext) + point_pos - cur_pos) - cur_pos_ext);
                it_pcl->x = p_compensate(0);
                it_pcl->y = p_compensate(1);
                it_pcl->z = p_compensate(2);

                if (it_pcl == out->points.begin())
                    break;
            }
        }
    }

    bool IMUProcessor::operator()(const MeasureGroup &meas, PointCloudXYZI::Ptr &out)
    {
        if (!init_flag_)
        {
            init(meas);
            return false;
        }
        undistortPointcloud(meas, out);//对点云进行畸变矫正
        return true;
    }

    void IMUProcessor::reset()
    {
        init_count_ = 0;
        init_flag_ = false;
        mean_acc_ = Eigen::Vector3d::Zero();
        mean_gyro_ = Eigen::Vector3d::Zero();
    }
} // namespace fastlio
