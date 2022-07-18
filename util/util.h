#ifndef UTIL_H
#define UTIL_H

// #include <cstdio>
#include <algorithm>
#include <vector>
#include <Eigen/Core>


template<typename T>
void erase_indices(std::vector<T>& data, std::vector<int>& indicesToDelete)
{
    if(indicesToDelete.empty())
        return;
    
    //assumes indices are sorted with descending order
    for(int i=0; i<indicesToDelete.size(); i++){
        data.erase(data.begin()+indicesToDelete[i]);
    }
}

template<typename T>
void erase_indices(std::vector<T,Eigen::aligned_allocator<T> >& data, std::vector<int>& indicesToDelete)
{
    if(indicesToDelete.empty())
        return;
    //assumes indices are sorted with descending order
    for(int i=0; i<indicesToDelete.size(); i++){
        data.erase(data.begin()+indicesToDelete[i]);
    }
}

// template<typename T>
// std::vector<T> erase_indices(const std::vector<T>& data, std::vector<int>& indicesToDelete/* can't assume copy elision, don't pass-by-value */)
// {
//     if(indicesToDelete.empty())
//         return data;

//     std::vector<T> ret;
//     ret.reserve(data.size() - indicesToDelete.size());

//     std::sort(indicesToDelete.begin(), indicesToDelete.end());

//     // new we can assume there is at least 1 element to delete. copy blocks at a time.
//     typename std::vector<T>::const_iterator itBlockBegin = data.begin();
//     for(std::vector<int>::const_iterator it = indicesToDelete.begin(); it != indicesToDelete.end(); ++ it)
//     {
//         typename std::vector<T>::const_iterator itBlockEnd = data.begin() + *it;
//         if(itBlockBegin != itBlockEnd)
//         {
//             std::copy(itBlockBegin, itBlockEnd, std::back_inserter(ret));
//         }
//         itBlockBegin = itBlockEnd + 1;
//     }

//     // copy last block.
//     if(itBlockBegin != data.end())
//     {
//         std::copy(itBlockBegin, data.end(), std::back_inserter(ret));
//     }

//     return ret;
// }

// template<typename T>
// std::vector<T,Eigen::aligned_allocator<T> > erase_indices(const std::vector<T,Eigen::aligned_allocator<T> >& data, std::vector<int>& indicesToDelete/* can't assume copy elision, don't pass-by-value */)
// {
//     if(indicesToDelete.empty())
//         return data;

//     std::vector<T,Eigen::aligned_allocator<T> > ret;
//     ret.reserve(data.size() - indicesToDelete.size());

//     std::sort(indicesToDelete.begin(), indicesToDelete.end());

//     // new we can assume there is at least 1 element to delete. copy blocks at a time.
//     typename std::vector<T,Eigen::aligned_allocator<T> >::const_iterator itBlockBegin = data.begin();
//     for(std::vector<int>::const_iterator it = indicesToDelete.begin(); it != indicesToDelete.end(); ++ it)
//     {
//         typename std::vector<T,Eigen::aligned_allocator<T> >::const_iterator itBlockEnd = data.begin() + *it;
//         if(itBlockBegin != itBlockEnd)
//         {
//             std::copy(itBlockBegin, itBlockEnd, std::back_inserter(ret));
//         }
//         itBlockBegin = itBlockEnd + 1;
//     }

//     // copy last block.
//     if(itBlockBegin != data.end())
//     {
//         std::copy(itBlockBegin, data.end(), std::back_inserter(ret));
//     }

//     return ret;
// }

#endif