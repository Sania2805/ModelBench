import { useCallback, useEffect, useRef, useState } from 'react'
import { getErrorMessage } from '../services/api'

export function useAsync(asyncFn, deps = [], { immediate = true } = {}) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(immediate)
  const [error, setError] = useState(null)
  const latestCall = useRef(0)

  const run = useCallback(asyncFn, deps)

  const reload = useCallback(async () => {
    const callId = ++latestCall.current
    setLoading(true)
    setError(null)
    try {
      const result = await run()
      if (callId === latestCall.current) setData(result)
      return result
    } catch (err) {
      if (callId === latestCall.current) setError(getErrorMessage(err))
      return null
    } finally {
      if (callId === latestCall.current) setLoading(false)
    }
  }, [run])

  useEffect(() => {
    if (immediate) reload()
  }, [reload, immediate])

  return { data, loading, error, reload, setData }
}
