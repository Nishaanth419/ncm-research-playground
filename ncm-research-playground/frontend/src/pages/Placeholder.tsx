import { Construction } from 'lucide-react'

import { EmptyState, PageHeader } from '@/components/common'

/** Honest placeholder for sections that land in later milestones. */
export function Placeholder({ title, description, milestone, detail }: {
  title: string
  description: string
  milestone: number
  detail: string
}) {
  return (
    <>
      <PageHeader title={title} description={description} />
      <div className="p-4 sm:p-6 lg:p-8">
        <EmptyState icon={Construction} title={`Not built yet: Milestone ${milestone}`}>
          {detail}
        </EmptyState>
      </div>
    </>
  )
}
