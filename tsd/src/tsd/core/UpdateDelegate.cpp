// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/UpdateDelegate.hpp"
// std
#include <algorithm>

namespace tsd {

size_t MultiUpdateDelegate::size() const
{
  return m_delegates.size();
}

void MultiUpdateDelegate::clear()
{
  m_delegates.clear();
}

void MultiUpdateDelegate::erase(const BaseUpdateDelegate *d)
{
  m_delegates.erase(std::remove_if(m_delegates.begin(),
                        m_delegates.end(),
                        [&](auto &ud) { return ud.get() == d; }),
      m_delegates.end());
}

const BaseUpdateDelegate *MultiUpdateDelegate::operator[](size_t i) const
{
  return m_delegates[i].get();
}

void MultiUpdateDelegate::signalObjectAdded(const Object *o)
{
  for (auto &d : m_delegates)
    d->signalObjectAdded(o);
}

void MultiUpdateDelegate::signalParameterUpdated(
    const Object *o, const Parameter *p)
{
  for (auto &d : m_delegates)
    d->signalParameterUpdated(o, p);
}

void MultiUpdateDelegate::signalParameterRemoved(
    const Object *o, const Parameter *p)
{
  for (auto &d : m_delegates)
    d->signalParameterRemoved(o, p);
}

void MultiUpdateDelegate::signalArrayMapped(const Array *a)
{
  for (auto &d : m_delegates)
    d->signalArrayMapped(a);
}

void MultiUpdateDelegate::signalArrayUnmapped(const Array *a)
{
  for (auto &d : m_delegates)
    d->signalArrayUnmapped(a);
}

void MultiUpdateDelegate::signalObjectRemoved(const Object *o)
{
  for (auto &d : m_delegates)
    d->signalObjectRemoved(o);
}

void MultiUpdateDelegate::signalRemoveAllObjects()
{
  for (auto &d : m_delegates)
    d->signalRemoveAllObjects();
}

void MultiUpdateDelegate::signalInstanceStructureChanged()
{
  for (auto &d : m_delegates)
    d->signalInstanceStructureChanged();
}

void MultiUpdateDelegate::signalObjectFilteringChanged()
{
  for (auto &d : m_delegates)
    d->signalObjectFilteringChanged();
}

} // namespace tsd
